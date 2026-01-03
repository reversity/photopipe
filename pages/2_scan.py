"""
Scan Page - Simple scanning that just works.
"""

import streamlit as st
import subprocess
import shutil
import re
from pathlib import Path
from datetime import datetime

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import Batch, PhotoPair, PhotoStatus
from photopipe.file_manager import generate_thumbnail


st.set_page_config(page_title="Scan - PhotoPipe", page_icon="📷", layout="wide")

SCANNER_DEVICE = "epsonds:net:192.168.1.62"


def init_session_state():
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()
    if "next_seq" not in st.session_state:
        st.session_state.next_seq = 1
    if "photos_scanned" not in st.session_state:
        st.session_state.photos_scanned = 0


def get_batch():
    db = st.session_state.db
    batches = db.get_all_batches()
    if not batches:
        st.warning("Create a batch first.")
        if st.button("Go to Batch Setup"):
            st.switch_page("pages/1_batch_setup.py")
        return None

    names = [b.name for b in batches]
    selected = st.selectbox("Batch", names, label_visibility="collapsed")
    return next((b for b in batches if b.name == selected), None)


def auto_rotate_image(path: Path):
    """Auto-rotate image based on dimensions (landscape photos should be horizontal)."""
    try:
        from PIL import Image
        img = Image.open(path)
        # If image is taller than wide, rotate 90 degrees
        if img.height > img.width:
            img = img.rotate(-90, expand=True)
            img.save(path, quality=95)
    except Exception:
        pass  # Skip rotation on error


def calculate_photo_date(batch: Batch, sequence: int, total: int):
    """Calculate a suggested date for a photo based on batch range and sequence."""
    if not batch.date_start:
        return None

    if not batch.date_end or batch.date_start == batch.date_end:
        return batch.date_start

    # Spread photos evenly across the date range
    from datetime import timedelta
    days_span = (batch.date_end - batch.date_start).days
    if total <= 1:
        return batch.date_start

    day_offset = int((sequence - 1) * days_span / (total - 1)) if total > 1 else 0
    return batch.date_start + timedelta(days=day_offset)


def scan_photos(batch: Batch, resolution: int, duplex: bool):
    """Scan photos and directly add them to the database."""
    db = st.session_state.db

    # Ensure input folder exists
    input_folder = Path.home() / "Pictures" / "Scanner_Input"
    input_folder.mkdir(parents=True, exist_ok=True)

    # Get next sequence number from database
    existing = db.get_photos_by_batch(batch.id)
    if existing:
        next_seq = max(p.sequence_num for p in existing) + 1
    else:
        next_seq = 1

    # Build scanimage command
    source = "ADF Duplex" if duplex else "ADF Front"
    batch_prefix = f"scan_{datetime.now().strftime('%H%M%S')}"
    batch_pattern = str(input_folder / f"{batch_prefix}_%04d.jpg")

    cmd = [
        "scanimage",
        "-d", SCANNER_DEVICE,
        "--resolution", str(resolution),
        "--mode", "color",
        "--format", "jpeg",
        "--source", source,
        "--adf-crp=yes",  # Auto-crop to photo edges
        "--adf-skew=yes",  # Correct skew
        f"--batch={batch_pattern}",
    ]

    # Run scanner
    status = st.empty()
    status.info("🔄 Scanning... Place photos in the scanner.")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env={**subprocess.os.environ, "PATH": "/opt/homebrew/bin:" + subprocess.os.environ.get("PATH", "")},
        )

        stderr = result.stderr.lower() if result.stderr else ""
        stdout = result.stdout if result.stdout else ""

        # Debug: show what happened
        st.caption(f"Scanner output: {result.returncode}")

        # Find scanned files
        scanned = sorted(input_folder.glob(f"{batch_prefix}_*.jpg"))

        if not scanned:
            status.warning(f"No photos found. Scanner said: {result.stderr[:200] if result.stderr else 'nothing'}")
            return 0

        # Process files: pair fronts and backs, add to database
        photos_added = 0
        total_photos = len(scanned) // 2 if duplex else len(scanned)

        if duplex:
            # In duplex mode, files alternate: front, back, front, back
            for i in range(0, len(scanned), 2):
                front_temp = scanned[i]
                back_temp = scanned[i + 1] if i + 1 < len(scanned) else None

                # Rename to permanent names
                seq = next_seq + photos_added
                front_path = input_folder / f"photo_{seq:04d}.jpg"
                back_path = input_folder / f"photo_{seq:04d}_b.jpg" if back_temp else None

                shutil.move(front_temp, front_path)
                if back_temp and back_path:
                    shutil.move(back_temp, back_path)

                # Auto-rotate front
                auto_rotate_image(front_path)

                # Calculate suggested date
                suggested_date = calculate_photo_date(batch, seq, next_seq + total_photos - 1)

                # Add to database
                photo = PhotoPair(
                    batch_id=batch.id,
                    sequence_num=seq,
                    front_path=front_path,
                    back_path=back_path,
                    status=PhotoStatus.INGESTED,
                    extracted_date=suggested_date,  # Set initial suggested date
                )
                db.create_photo(photo)
                photos_added += 1
        else:
            # Single-sided
            for front_temp in scanned:
                seq = next_seq + photos_added
                front_path = input_folder / f"photo_{seq:04d}.jpg"
                shutil.move(front_temp, front_path)

                # Auto-rotate
                auto_rotate_image(front_path)

                # Calculate suggested date
                suggested_date = calculate_photo_date(batch, seq, next_seq + total_photos - 1)

                photo = PhotoPair(
                    batch_id=batch.id,
                    sequence_num=seq,
                    front_path=front_path,
                    back_path=None,
                    status=PhotoStatus.INGESTED,
                    extracted_date=suggested_date,
                )
                db.create_photo(photo)
                photos_added += 1

        # Check for jam
        if "jam" in stderr:
            status.warning(f"⚠️ Paper jam after {photos_added} photos. Clear jam and scan again to continue.")
        else:
            status.success(f"✅ Scanned {photos_added} photos")

        st.session_state.photos_scanned += photos_added
        return photos_added

    except subprocess.TimeoutExpired:
        status.error("Scanning timed out")
        return 0
    except Exception as e:
        status.error(f"Scan error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return 0


def show_photos(batch: Batch):
    db = st.session_state.db
    photos = db.get_photos_by_batch(batch.id)

    if not photos:
        st.info("No photos in batch. Scan some photos above.")
        return

    st.subheader(f"📸 {len(photos)} Photos")

    # Grid
    cols = st.columns(6)
    for i, photo in enumerate(photos[:18]):
        with cols[i % 6]:
            try:
                if photo.front_path and photo.front_path.exists():
                    thumb = generate_thumbnail(photo.front_path, size=120)
                    st.image(thumb, use_container_width=True)
                st.caption(f"#{photo.sequence_num}")
            except:
                st.caption(f"#{photo.sequence_num}")

    if len(photos) > 18:
        st.caption(f"+ {len(photos) - 18} more")

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Review Metadata", type="primary", use_container_width=True):
            st.switch_page("pages/3_review.py")
    with col2:
        if st.button("📤 Export", use_container_width=True):
            st.switch_page("pages/4_finalize.py")


def main():
    init_session_state()
    st.title("📷 Scan")

    batch = get_batch()
    if not batch:
        return

    st.caption(f"📅 {batch.get_date_range_str() or 'No date'} | 📍 {batch.location_description or 'No location'}")
    st.markdown("---")

    # Scan controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        resolution = st.selectbox("DPI", [300, 600, 1200], index=1)
    with col2:
        duplex = st.checkbox("Scan backs", value=True)
    with col3:
        if st.button("📷 Scan Photos", type="primary", use_container_width=True):
            scan_photos(batch, resolution, duplex)
            st.rerun()

    if st.session_state.photos_scanned > 0:
        st.success(f"✓ {st.session_state.photos_scanned} photos scanned this session")
        if st.button("Clear count"):
            st.session_state.photos_scanned = 0
            st.rerun()

    st.markdown("---")
    show_photos(batch)


if __name__ == "__main__":
    main()
