"""
Finalize & Export Page - Export photos with embedded metadata.
"""

import streamlit as st
from pathlib import Path

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import Batch, PhotoStatus, DateSource
from photopipe.file_manager import (
    finalize_batch,
    generate_output_folder,
    sanitize_filename,
)
from photopipe.metadata import check_exiftool_installed


st.set_page_config(page_title="Finalize - PhotoPipe", page_icon="✅", layout="wide")


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None

    if "finalizing" not in st.session_state:
        st.session_state.finalizing = False


def batch_selector():
    """Render batch selector."""
    db = st.session_state.db
    batches = db.get_all_batches()

    # Filter to batches with photos
    batches = [b for b in batches if db.get_batch_photo_count(b.id) > 0]

    if not batches:
        st.warning("No batches with photos found.")
        return None

    batch_options = {b.name: b.id for b in batches}

    selected_name = st.selectbox(
        "Select Batch to Finalize",
        options=list(batch_options.keys()),
    )

    st.session_state.current_batch_id = batch_options[selected_name]
    return db.get_batch(st.session_state.current_batch_id)


def batch_summary(batch: Batch):
    """Display batch summary statistics."""
    st.subheader("📊 Batch Summary")

    db = st.session_state.db
    stats = db.get_batch_stats(batch.id)
    photos = db.get_photos_by_batch(batch.id)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Photos", stats["total"])

    with col2:
        reviewed = stats["by_status"].get("reviewed", 0) + stats["by_status"].get("finalized", 0)
        st.metric("Reviewed", reviewed)

    with col3:
        st.metric("Needs Review", stats["needs_review"])

    with col4:
        finalized = stats["by_status"].get("finalized", 0)
        st.metric("Already Finalized", finalized)

    st.markdown("---")

    # Date sources breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Date Sources:**")
        for source, count in stats["by_date_source"].items():
            pct = (count / stats["total"] * 100) if stats["total"] > 0 else 0
            source_label = source.replace("_", " ").title() if source != "none" else "No Date"
            st.write(f"- {source_label}: {count} ({pct:.0f}%)")

    with col2:
        st.write("**Batch Metadata:**")
        st.write(f"- **Date Range:** {batch.get_date_range_str()}")
        if batch.location_description:
            st.write(f"- **Location:** {batch.location_description}")
        if batch.people:
            st.write(f"- **People:** {', '.join(batch.people[:5])}{'...' if len(batch.people) > 5 else ''}")

    # Warnings
    if stats["needs_review"] > 0:
        st.warning(f"""
        ⚠️ **{stats["needs_review"]} photos still need review.**

        These photos will be skipped during finalization unless you:
        1. Review and approve them in the Review Queue
        2. Enable "Auto-approve high confidence" below
        """)

    return stats


def output_preview(batch: Batch):
    """Preview output folder structure."""
    st.subheader("📁 Output Preview")

    config = get_config()
    db = st.session_state.db

    # Calculate output folder
    output_folder = generate_output_folder(batch)

    st.write("**Output Location:**")
    st.code(str(output_folder))

    # Show folder structure
    st.write("**Folder Structure:**")
    structure = f"""
{config.paths.output_folder}/
├── {output_folder.relative_to(config.paths.output_folder)}/
│   ├── {batch.date_start.strftime('%Y-%m-%d') if batch.date_start else 'YYYY-MM-DD'}_{sanitize_filename(batch.name)}_0001_front.jpg
│   ├── {batch.date_start.strftime('%Y-%m-%d') if batch.date_start else 'YYYY-MM-DD'}_{sanitize_filename(batch.name)}_0001_back.jpg
│   ├── ...
│   └── _batch_report.json
└── _archive/
    └── {sanitize_filename(batch.name)}/
        └── (original files)
    """
    st.code(structure)

    # Archive settings
    st.write("**Archive Settings:**")
    col1, col2 = st.columns(2)

    with col1:
        preserve = st.checkbox(
            "Preserve original files in archive",
            value=config.output.preserve_originals,
        )

    with col2:
        web_copies = st.checkbox(
            "Generate web-sized copies",
            value=config.output.generate_web_copies,
        )

    return preserve, web_copies


def finalize_controls(batch: Batch, stats: dict):
    """Render finalize controls."""
    st.subheader("🚀 Finalize Batch")

    db = st.session_state.db

    # Check prerequisites
    if not check_exiftool_installed():
        st.error("""
        ❌ **ExifTool is not installed.**

        Please install ExifTool to write metadata:
        ```bash
        brew install exiftool
        ```
        """)
        return

    ready_count = stats["by_status"].get("reviewed", 0)
    needs_review = stats["needs_review"]

    if ready_count == 0 and needs_review > 0:
        st.warning("No photos are ready for finalization. Please review photos first.")

    # Options
    col1, col2 = st.columns(2)

    with col1:
        auto_approve = st.checkbox(
            "Auto-approve high confidence dates",
            value=True,
            help="Automatically approve photos with high confidence OCR dates",
        )

    with col2:
        skip_uncertain = st.checkbox(
            "Skip photos without dates",
            value=False,
            help="Don't finalize photos that have no date (will use batch date otherwise)",
        )

    st.markdown("---")

    # Finalize button
    if st.button(
        "✨ Finalize Batch",
        type="primary",
        disabled=st.session_state.finalizing,
    ):
        st.session_state.finalizing = True

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Finalizing photo {current} of {total}...")

        try:
            with st.spinner("Finalizing batch..."):
                report = finalize_batch(
                    batch,
                    db,
                    auto_approve_high_confidence=auto_approve,
                    progress_callback=update_progress,
                )

            st.success(f"""
            ✅ **Batch Finalized Successfully!**

            - Photos processed: {report.photo_count}
            - Output folder: `{generate_output_folder(batch)}`
            """)

            # Show report summary
            with st.expander("📋 View Batch Report"):
                st.json({
                    "batch_name": report.batch_name,
                    "photo_count": report.photo_count,
                    "date_range": report.date_range,
                    "date_source_breakdown": report.date_source_breakdown,
                    "people_tagged": report.people_tagged,
                })

            # Update batch status
            batch.status = "complete"
            db.update_batch(batch)

        except Exception as e:
            st.error(f"❌ Finalization failed: {e}")

        st.session_state.finalizing = False


def completed_batches():
    """Show list of completed batches."""
    st.subheader("✅ Completed Batches")

    db = st.session_state.db
    batches = db.get_all_batches()
    completed = [b for b in batches if b.status == "complete"]

    if not completed:
        st.info("No completed batches yet.")
        return

    for batch in completed:
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.write(f"**{batch.name}**")
            st.caption(batch.get_date_range_str())

        with col2:
            stats = db.get_batch_stats(batch.id)
            st.write(f"📷 {stats['total']} photos")

        with col3:
            output_folder = generate_output_folder(batch)
            if output_folder.exists():
                if st.button("📂 Open", key=f"open_{batch.id}"):
                    import subprocess
                    subprocess.run(["open", str(output_folder)])

        st.markdown("---")


def main():
    """Main page."""
    init_session_state()

    st.title("✅ Finalize & Export")
    st.write("Write metadata to photos and organize into output folders.")

    # Check for ExifTool
    if not check_exiftool_installed():
        st.error("""
        **ExifTool Required**

        PhotoPipe uses ExifTool to write metadata to your photos.

        Install on Mac:
        ```bash
        brew install exiftool
        ```
        """)
        st.stop()

    # Batch selector
    batch = batch_selector()
    if not batch:
        # Show completed batches instead
        completed_batches()
        return

    st.markdown("---")

    # Batch summary
    stats = batch_summary(batch)

    st.markdown("---")

    # Output preview
    preserve, web_copies = output_preview(batch)

    st.markdown("---")

    # Finalize controls
    finalize_controls(batch, stats)

    st.markdown("---")

    # Show completed batches
    completed_batches()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Review"):
            st.switch_page("pages/3_review.py")
    with col2:
        if st.button("Start New Batch →"):
            st.switch_page("pages/1_batch_setup.py")


if __name__ == "__main__":
    main()
