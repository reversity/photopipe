"""
Scan Page - Simplified scanning with auto-ingest.
"""

import streamlit as st
from pathlib import Path

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import Batch, PhotoStatus
from photopipe.pairing import pair_scans
from photopipe.scanner import Scanner, check_sane_installed
from photopipe.file_manager import generate_thumbnail


st.set_page_config(page_title="Scan - PhotoPipe", page_icon="📷", layout="wide")

# Known scanner
SCANNER_DEVICE = "epsonds:net:192.168.1.62"


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "last_scanned_sequence" not in st.session_state:
        st.session_state.last_scanned_sequence = None
    if "scan_error" not in st.session_state:
        st.session_state.scan_error = None
    if "scanned_this_session" not in st.session_state:
        st.session_state.scanned_this_session = 0
    if "scanning" not in st.session_state:
        st.session_state.scanning = False


def get_current_batch():
    """Get or select current batch."""
    db = st.session_state.db
    batches = db.get_all_batches()

    if not batches:
        st.warning("No batches found. Create a batch first.")
        if st.button("Go to Batch Setup"):
            st.switch_page("pages/1_batch_setup.py")
        return None

    # Simple batch selector
    batch_names = [b.name for b in batches]
    selected = st.selectbox("Batch", batch_names, label_visibility="collapsed")

    for b in batches:
        if b.name == selected:
            return b
    return None


def scan_and_ingest(batch: Batch):
    """Scan photos and automatically ingest them."""
    db = st.session_state.db
    config = get_config()
    input_folder = batch.input_folder or config.paths.input_folder

    # Ensure folder exists
    input_folder.mkdir(parents=True, exist_ok=True)

    # Get next sequence number
    if st.session_state.last_scanned_sequence:
        next_seq = st.session_state.last_scanned_sequence + 1
    else:
        next_seq = db.get_next_sequence_num(batch.id)

    # Settings in a compact row
    col1, col2, col3 = st.columns(3)
    with col1:
        resolution = st.selectbox("DPI", [300, 600, 1200], index=1)
    with col2:
        duplex = st.checkbox("Scan backs", value=True)
    with col3:
        st.metric("Next #", next_seq)

    # Main scan button
    col1, col2 = st.columns(2)

    with col1:
        scan_clicked = st.button(
            "📷 Scan Photos" if not st.session_state.last_scanned_sequence else "📷 Scan More Photos",
            type="primary",
            disabled=st.session_state.scanning,
            use_container_width=True
        )

    with col2:
        if st.session_state.last_scanned_sequence:
            if st.button("✅ Done Scanning", use_container_width=True):
                st.session_state.last_scanned_sequence = None
                st.session_state.scanned_this_session = 0
                st.session_state.scan_error = None
                st.switch_page("pages/3_review.py")

    # Show session status
    if st.session_state.scanned_this_session > 0:
        st.success(f"✓ {st.session_state.scanned_this_session} photos scanned this session")

    if st.session_state.scan_error:
        st.warning(f"Last scan: {st.session_state.scan_error}")
        st.info("Clear the jam and click 'Scan More Photos' to continue.")

    # Execute scan
    if scan_clicked:
        st.session_state.scanning = True
        st.session_state.scan_error = None

        progress = st.progress(0)
        status = st.empty()
        scanned_count = 0

        def update_progress(count, total):
            nonlocal scanned_count
            scanned_count = count
            status.text(f"Scanning photo #{next_seq + count - 1}...")
            st.session_state.last_scanned_sequence = next_seq + count - 1

        try:
            scanner = Scanner(device_name=SCANNER_DEVICE)

            results = scanner.scan_batch(
                output_folder=input_folder,
                name_prefix=batch.name.replace(" ", "_"),
                start_sequence=next_seq,
                resolution=resolution,
                duplex=duplex,
                mode="color",
                progress_callback=update_progress,
            )

            if results:
                st.session_state.last_scanned_sequence = results[-1].sequence_num
                st.session_state.scanned_this_session += len(results)

                # Auto-ingest
                status.text("Importing photos...")
                new_photos = pair_scans(input_folder, batch, db)

                progress.progress(100)
                status.empty()
                st.success(f"✓ Scanned and imported {len(results)} photos")
                st.rerun()
            else:
                status.empty()
                st.info("No photos scanned. Load photos in the scanner and try again.")

        except Exception as e:
            error_msg = str(e)
            st.session_state.scan_error = error_msg

            # Try to save any photos that were scanned
            if scanned_count > 0:
                st.session_state.last_scanned_sequence = next_seq + scanned_count - 1
                st.session_state.scanned_this_session += scanned_count

                try:
                    new_photos = pair_scans(input_folder, batch, db)
                    if new_photos:
                        st.success(f"✓ Saved {len(new_photos)} photos before the interruption")
                except Exception:
                    pass

            st.rerun()

        finally:
            st.session_state.scanning = False


def show_batch_photos(batch: Batch):
    """Show photos in the batch."""
    db = st.session_state.db
    photos = db.get_photos_by_batch(batch.id)

    if not photos:
        st.info("No photos in batch yet. Scan some photos above.")
        return

    st.subheader(f"📸 {len(photos)} Photos in Batch")

    # Summary
    with_dates = len([p for p in photos if p.extracted_date])
    needs_review = len([p for p in photos if p.needs_review])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", len(photos))
    with col2:
        st.metric("Have Dates", with_dates)
    with col3:
        if needs_review > 0:
            st.metric("Need Review", needs_review, delta=f"-{needs_review}", delta_color="inverse")
        else:
            st.metric("Need Review", 0)

    # Photo grid
    cols = st.columns(6)
    for i, photo in enumerate(photos[:24]):  # Show first 24
        with cols[i % 6]:
            try:
                if photo.front_path and Path(photo.front_path).exists():
                    thumb = generate_thumbnail(Path(photo.front_path), size=150)
                    st.image(thumb, use_container_width=True)

                    # Status indicator
                    if photo.needs_review:
                        st.caption(f"#{photo.sequence_num} ⚠️")
                    elif photo.extracted_date:
                        st.caption(f"#{photo.sequence_num} ✓")
                    else:
                        st.caption(f"#{photo.sequence_num}")
            except Exception:
                st.caption(f"#{photo.sequence_num}")

    if len(photos) > 24:
        st.caption(f"... and {len(photos) - 24} more")

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Review & Edit Metadata", type="primary", use_container_width=True):
            st.switch_page("pages/3_review.py")
    with col2:
        if st.button("📤 Export Photos", use_container_width=True):
            st.switch_page("pages/4_finalize.py")


def main():
    init_session_state()

    st.title("📷 Scan Photos")

    # Check scanner
    if not check_sane_installed():
        st.error("Scanner software (SANE) not installed. Run: `brew install sane-backends`")
        return

    # Batch selector
    batch = get_current_batch()
    if not batch:
        return

    # Show batch info
    st.caption(f"📍 {batch.location_description or 'No location'} | 📅 {batch.get_date_range_str() or 'No date'}")

    st.markdown("---")

    # Scanning section
    scan_and_ingest(batch)

    st.markdown("---")

    # Photos in batch
    show_batch_photos(batch)


if __name__ == "__main__":
    main()
