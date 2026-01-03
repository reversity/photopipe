"""
Scan Ingestion Page - Import and pair scans from scanner.
"""

import streamlit as st
from pathlib import Path
import time

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import Batch, PhotoStatus
from photopipe.pairing import pair_scans, get_pairing_summary
from photopipe.ocr import process_batch_ocr
from photopipe.ai_dating import (
    estimate_batch_date_with_ai,
    apply_ai_date_to_batch,
    is_ai_dating_available,
)
from photopipe.file_manager import generate_thumbnail
from photopipe.scanner import (
    Scanner,
    list_scanners,
    find_fastfoto,
    check_sane_installed,
    scan_photos_to_batch,
)


st.set_page_config(page_title="Scan Ingest - PhotoPipe", page_icon="📥", layout="wide")


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None

    if "watcher_running" not in st.session_state:
        st.session_state.watcher_running = False

    if "processing" not in st.session_state:
        st.session_state.processing = False


def batch_selector():
    """Render batch selector."""
    db = st.session_state.db
    batches = db.get_all_batches()

    if not batches:
        st.warning("No batches found. Please create a batch first.")
        if st.button("Go to Batch Setup"):
            st.switch_page("pages/1_batch_setup.py")
        return None

    batch_options = {b.name: b.id for b in batches}
    current_name = None

    if st.session_state.current_batch_id:
        for name, bid in batch_options.items():
            if bid == st.session_state.current_batch_id:
                current_name = name
                break

    selected_name = st.selectbox(
        "Select Batch",
        options=list(batch_options.keys()),
        index=list(batch_options.keys()).index(current_name) if current_name else 0,
    )

    st.session_state.current_batch_id = batch_options[selected_name]
    return db.get_batch(st.session_state.current_batch_id)


def folder_preview(batch: Batch):
    """Preview what's in the input folder."""
    st.subheader("📂 Input Folder Preview")

    input_folder = batch.input_folder or get_config().paths.input_folder

    if not input_folder.exists():
        st.warning(f"Input folder does not exist: {input_folder}")
        if st.button("Create Folder"):
            input_folder.mkdir(parents=True, exist_ok=True)
            st.success("Folder created!")
            st.rerun()
        return

    # Get pairing summary
    summary = get_pairing_summary(input_folder)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", summary["total_images"])
    with col2:
        st.metric("Complete Pairs", summary["complete_pairs"])
    with col3:
        st.metric("Fronts Only", summary["fronts_without_backs"])
    with col4:
        st.metric("Orphaned Backs", summary["orphaned_backs"])

    if summary["orphaned_backs"] > 0:
        st.warning(f"⚠️ Found {summary['orphaned_backs']} back images without matching fronts")

    # Show thumbnails of first few photos
    if summary["pairs"]:
        st.write("**Preview of photos to ingest:**")

        cols = st.columns(5)
        for i, pair in enumerate(summary["pairs"][:10]):
            with cols[i % 5]:
                try:
                    thumb = generate_thumbnail(Path(pair["front"]))
                    st.image(thumb, caption=f"#{pair['sequence']}", use_container_width=True)
                except Exception:
                    st.write(f"#{pair['sequence']}")

        if len(summary["pairs"]) > 10:
            st.caption(f"... and {len(summary['pairs']) - 10} more")


def ingest_photos(batch: Batch):
    """Ingest photos from input folder."""
    st.subheader("📥 Ingest Photos")

    db = st.session_state.db
    input_folder = batch.input_folder or get_config().paths.input_folder

    # Show current batch stats
    stats = db.get_batch_stats(batch.id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Photos in Batch", stats["total"])
    with col2:
        ingested = stats["by_status"].get("ingested", 0)
        st.metric("Awaiting OCR", ingested)
    with col3:
        processed = stats["by_status"].get("processed", 0)
        st.metric("Processed", processed)

    st.markdown("---")

    # Ingest button
    if st.button("🔍 Scan & Ingest Photos", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True

        with st.spinner("Scanning input folder and pairing images..."):
            new_photos = pair_scans(input_folder, batch, db)

        if new_photos:
            st.success(f"✅ Ingested {len(new_photos)} new photo pairs!")
            st.rerun()
        else:
            st.info("No new photos found to ingest.")

        st.session_state.processing = False


def process_ocr(batch: Batch):
    """Process OCR on ingested photos."""
    st.subheader("🔤 OCR Processing")

    with st.expander("💡 How OCR works", expanded=False):
        st.markdown("""
        **OCR (Optical Character Recognition)** reads text from the back of your photos.

        Many old photos have dates, names, or notes written on the back. PhotoPipe:
        1. Scans the back image for handwritten or printed text
        2. Looks for date patterns (e.g., "June 1985", "6/15/85")
        3. Extracts and saves any dates found

        **Tips for better results:**
        - Use clear, dark ink when writing on photos
        - Ensure the scanner glass is clean
        - Photos with low-confidence OCR will be flagged for review
        """)

    db = st.session_state.db

    # Get photos needing OCR
    photos = db.get_photos_by_batch(batch.id, status=PhotoStatus.INGESTED)

    if not photos:
        st.info("No photos awaiting OCR processing.")
        return

    st.write(f"**{len(photos)} photos** ready for OCR processing")

    if st.button("🚀 Run OCR", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Processing photo {current} of {total}...")

        with st.spinner("Running OCR..."):
            processed = process_batch_ocr(
                batch.id,
                db,
                max_workers=4,
                progress_callback=update_progress,
            )

        # Count results
        dates_found = len([p for p in processed if p.extracted_date])
        needs_review = len([p for p in processed if p.needs_review])

        st.success(f"""
        ✅ OCR Complete!
        - Processed: {len(processed)} photos
        - Dates found: {dates_found}
        - Needs review: {needs_review}
        """)

        # Update batch status
        batch.status = "review" if needs_review > 0 else "processing"
        db.update_batch(batch)

        st.session_state.processing = False
        st.rerun()


def ai_date_estimation(batch: Batch):
    """AI-assisted date estimation section."""
    st.subheader("🤖 AI Date Estimation")

    with st.expander("💡 How AI dating works", expanded=False):
        st.markdown("""
        **AI dating** uses Claude to analyze visual clues in your photos.

        The AI looks at:
        - 👔 **Clothing & fashion** - styles change by decade
        - 💇 **Hairstyles** - very era-specific
        - 🚗 **Vehicles** - car models are great date markers
        - 📺 **Technology** - TVs, phones, computers
        - 🏠 **Interior design** - furniture, wallpaper, decor

        **How it works:**
        1. PhotoPipe selects a few representative photos
        2. Sends them to Claude for analysis
        3. AI estimates a likely date range with confidence level
        4. You can apply this date to all undated photos in the batch

        **Note:** AI dating requires an Anthropic API key (set in Settings).
        """)

    if not is_ai_dating_available():
        st.info("""
        AI dating is not available. To enable:
        1. Set your `ANTHROPIC_API_KEY` environment variable
        2. Ensure the `anthropic` package is installed
        """)
        return

    db = st.session_state.db

    # Count photos without dates
    photos = db.get_photos_by_batch(batch.id)
    photos_without_dates = [p for p in photos if p.extracted_date is None]

    if not photos_without_dates:
        st.info("All photos have dates. AI estimation not needed.")
        return

    st.write(f"**{len(photos_without_dates)} photos** don't have dates extracted.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        AI dating analyzes representative photos from your batch to estimate
        when they were taken based on:
        - Clothing and fashion styles
        - Hairstyles
        - Visible technology (cars, TVs, phones)
        - Interior/exterior details
        """)

    with col2:
        config = get_config()
        st.caption(f"Model: {config.ai_dating.model}")
        st.caption(f"Samples: {config.ai_dating.max_samples_per_batch}")

    if st.button("🔮 Estimate Dates with AI", type="secondary", disabled=st.session_state.processing):
        st.session_state.processing = True

        with st.spinner("Analyzing photos with AI..."):
            estimate = estimate_batch_date_with_ai(batch, photos_without_dates, db)

        if estimate:
            st.success(f"""
            ✅ AI Analysis Complete!
            - Estimated year: {estimate.year or f"{estimate.year_range[0]}-{estimate.year_range[1]}" if estimate.year_range else "Unknown"}
            - Confidence: {estimate.confidence.value}
            """)

            with st.expander("View Evidence"):
                for evidence in estimate.evidence:
                    st.write(f"- {evidence}")
                st.write(f"**Reasoning:** {estimate.reasoning}")

            if st.button("Apply AI Date to Photos"):
                updated = apply_ai_date_to_batch(batch, estimate, photos_without_dates, db)
                st.success(f"Applied AI date to {updated} photos")
                st.rerun()
        else:
            st.warning("Could not estimate date from photos")

        st.session_state.processing = False


def scanner_control(batch: Batch):
    """Scanner control section for directly scanning photos."""
    st.subheader("📷 Scanner Control")

    # Initialize scanner session state
    if "last_scanned_sequence" not in st.session_state:
        st.session_state.last_scanned_sequence = None
    if "scan_error" not in st.session_state:
        st.session_state.scan_error = None
    if "scanned_this_session" not in st.session_state:
        st.session_state.scanned_this_session = 0

    # Check SANE availability
    if not check_sane_installed():
        st.error("""
        **SANE not installed**

        Scanner control requires SANE (Scanner Access Now Easy).

        On macOS, you may need to use the Epson FastFoto software directly,
        or install SANE via Homebrew:
        ```bash
        brew install sane-backends
        ```
        """)
        return

    config = get_config()

    # Try to list available scanners, but don't block if detection fails
    devices = []
    with st.spinner("Detecting scanners..."):
        try:
            devices = list_scanners()
        except Exception:
            pass

    # Build device options - include detected devices plus manual entry option
    device_options = {}
    if devices:
        device_options = {str(d): d.name for d in devices}

    # Always add the known network scanner as an option (detection is unreliable for network scanners)
    KNOWN_NETWORK_SCANNER = "epsonds:net:192.168.1.62"
    device_options["Epson FF-680W (WiFi)"] = KNOWN_NETWORK_SCANNER

    # Also check if there's a configured device in settings
    if config.scanner.device and config.scanner.device not in device_options.values():
        device_options[f"Configured: {config.scanner.device}"] = config.scanner.device

    if not device_options:
        st.warning("""
        **No scanners available**

        Make sure your Epson FastFoto FF-680W is:
        1. Connected via USB or WiFi
        2. Powered on
        """)
        return

    # Scanner selection
    selected_device = st.selectbox(
        "Select Scanner",
        options=list(device_options.keys()),
        index=0,
    )

    device_name = device_options[selected_device]

    # Show connection status
    if "net:" in device_name:
        st.caption(f"📡 Network scanner: `{device_name}`")

    # Scan settings
    st.write("**Scan Settings**")
    col1, col2, col3 = st.columns(3)

    with col1:
        resolution = st.selectbox(
            "Resolution (DPI)",
            options=[300, 600, 1200],
            index=1,  # Default 600
        )

    with col2:
        duplex = st.checkbox("Scan Back Side (Duplex)", value=True)

    with col3:
        mode = st.selectbox(
            "Color Mode",
            options=["color", "gray"],
            index=0,
        )

    st.markdown("---")

    # Scan status and jam recovery
    db = st.session_state.db
    input_folder = batch.input_folder or config.paths.input_folder
    default_start_sequence = db.get_next_sequence_num(batch.id)

    # Show jam recovery UI if there was an error or previous scan
    if st.session_state.scan_error or st.session_state.last_scanned_sequence:
        st.warning("⚠️ **Jam Recovery Mode**")

        if st.session_state.scan_error:
            st.error(f"Last error: {st.session_state.scan_error}")

        if st.session_state.last_scanned_sequence:
            st.info(f"""
            **Last successful scan:** #{st.session_state.last_scanned_sequence}
            **Photos scanned this session:** {st.session_state.scanned_this_session}

            To resume after clearing a jam:
            1. Remove jammed photo from scanner
            2. Note which photo jammed (check the last scanned number above)
            3. Click "Resume Scanning" to continue from the next photo
            """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Resume Scanning", type="primary"):
                st.session_state.scan_error = None
                # Will use the resume sequence below
        with col2:
            if st.button("🔁 Start Fresh"):
                st.session_state.last_scanned_sequence = None
                st.session_state.scan_error = None
                st.session_state.scanned_this_session = 0
                st.rerun()

    # Determine starting sequence
    if st.session_state.last_scanned_sequence:
        resume_sequence = st.session_state.last_scanned_sequence + 1
    else:
        resume_sequence = default_start_sequence

    st.write(f"**Output folder:** `{input_folder}`")

    # Allow manual sequence override for recovery
    col1, col2 = st.columns([2, 1])
    with col1:
        start_sequence = st.number_input(
            "Starting sequence number",
            min_value=1,
            value=resume_sequence,
            help="Adjust this if you need to re-scan specific photos or skip ahead",
        )
    with col2:
        st.write("")  # Spacer
        st.caption(f"Default: {default_start_sequence}")

    # Scan button
    button_label = "🔄 Resume Scanning" if st.session_state.last_scanned_sequence else "🚀 Start Scanning"

    if st.button(button_label, type="primary", disabled=st.session_state.processing, key="scan_btn"):
        st.session_state.processing = True
        st.session_state.scan_error = None

        progress_bar = st.progress(0)
        status_text = st.empty()
        scanned_count = 0

        def update_progress(count, total):
            nonlocal scanned_count
            scanned_count = count
            if total:
                progress_bar.progress(count / total)
            status_text.text(f"Scanned {count} photos... (Current: #{start_sequence + count - 1})")
            # Update session state so we can recover from jams
            st.session_state.last_scanned_sequence = start_sequence + count - 1
            st.session_state.scanned_this_session += 1

        try:
            scanner = Scanner(device_name=device_name)

            with st.spinner("Scanning photos... Place photos in the ADF and they will be scanned automatically."):
                results = scanner.scan_batch(
                    output_folder=input_folder,
                    name_prefix=batch.name,
                    start_sequence=start_sequence,
                    resolution=resolution,
                    duplex=duplex,
                    mode=mode,
                    progress_callback=update_progress,
                )

            if results:
                st.success(f"✅ Scanned {len(results)} photos!")
                st.session_state.last_scanned_sequence = results[-1].sequence_num

                # Auto-ingest the scanned photos
                with st.spinner("Ingesting scanned photos..."):
                    new_photos = pair_scans(input_folder, batch, db)

                st.success(f"✅ Ingested {len(new_photos)} photo pairs into batch!")

                # Clear jam recovery state on success
                st.session_state.scan_error = None
                st.rerun()
            else:
                st.info("No photos were scanned. Make sure photos are loaded in the ADF.")

        except Exception as e:
            error_msg = str(e)
            st.session_state.scan_error = error_msg

            # Check if any photos were scanned before the error
            if scanned_count > 0:
                st.warning(f"""
                ⚠️ **Scanning interrupted after {scanned_count} photos**

                Last successful scan: #{st.session_state.last_scanned_sequence}

                **To recover:**
                1. Clear the paper jam
                2. Remove the photo that jammed
                3. Click "Resume Scanning" to continue from #{st.session_state.last_scanned_sequence + 1}
                """)

                # Try to ingest what we got
                with st.spinner("Ingesting photos scanned before error..."):
                    try:
                        new_photos = pair_scans(input_folder, batch, db)
                        if new_photos:
                            st.success(f"✅ Saved {len(new_photos)} photos scanned before the jam")
                    except Exception:
                        pass
            else:
                st.error(f"Scanning failed: {error_msg}")

        st.session_state.processing = False

    # Show reset button if we have scan history
    if st.session_state.scanned_this_session > 0:
        st.markdown("---")
        st.caption(f"Session total: {st.session_state.scanned_this_session} photos scanned")
        if st.button("✅ Finished Scanning - Clear Session", key="clear_session"):
            st.session_state.last_scanned_sequence = None
            st.session_state.scan_error = None
            st.session_state.scanned_this_session = 0
            st.rerun()


def batch_progress(batch: Batch):
    """Show overall batch progress."""
    st.subheader("📊 Batch Progress")

    db = st.session_state.db
    stats = db.get_batch_stats(batch.id)

    if stats["total"] == 0:
        st.info("No photos in batch yet. Ingest some photos to get started!")
        return

    # Progress by status
    total = stats["total"]
    ingested = stats["by_status"].get("ingested", 0)
    processed = stats["by_status"].get("processed", 0)
    reviewed = stats["by_status"].get("reviewed", 0)
    finalized = stats["by_status"].get("finalized", 0)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Processing Status:**")
        st.progress(ingested / total if total > 0 else 0, text=f"Ingested: {ingested}")
        st.progress(processed / total if total > 0 else 0, text=f"OCR Complete: {processed}")
        st.progress(reviewed / total if total > 0 else 0, text=f"Reviewed: {reviewed}")
        st.progress(finalized / total if total > 0 else 0, text=f"Finalized: {finalized}")

    with col2:
        st.write("**Date Sources:**")
        for source, count in stats["by_date_source"].items():
            pct = (count / total * 100) if total > 0 else 0
            st.write(f"- {source.replace('_', ' ').title()}: {count} ({pct:.0f}%)")

        if stats["needs_review"] > 0:
            st.warning(f"⚠️ {stats['needs_review']} photos need review")


def main():
    """Main page."""
    init_session_state()

    st.title("📥 Scan Ingestion")
    st.write("Import scanned photos and run OCR processing.")

    # Batch selector
    batch = batch_selector()
    if not batch:
        return

    st.markdown("---")

    # Show batch info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Batch:** {batch.name}")
        st.write(f"**Date Range:** {batch.get_date_range_str()}")
    with col2:
        st.write(f"**Status:** {batch.status.title()}")
        if batch.location_description:
            st.write(f"**Location:** {batch.location_description}")

    st.markdown("---")

    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 Scanner", "📂 Preview", "📥 Ingest", "🔤 OCR", "🤖 AI Dating"])

    with tab1:
        scanner_control(batch)

    with tab2:
        folder_preview(batch)

    with tab3:
        ingest_photos(batch)

    with tab4:
        process_ocr(batch)

    with tab5:
        ai_date_estimation(batch)

    st.markdown("---")

    # Batch progress
    batch_progress(batch)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Batch Setup"):
            st.switch_page("pages/1_batch_setup.py")
    with col2:
        if st.button("Continue to Review →"):
            st.switch_page("pages/3_review.py")


if __name__ == "__main__":
    main()
