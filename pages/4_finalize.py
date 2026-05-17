"""
Finalize & Export Page - Export photos with embedded metadata.
"""

import subprocess
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


def get_photos_albums() -> list[str]:
    """Get list of existing album names from Photos app."""
    script = '''
    tell application "Photos"
        set albumNames to {}
        repeat with anAlbum in albums
            set end of albumNames to name of anAlbum
        end repeat
        return albumNames
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse comma-separated album names
            albums = [a.strip() for a in result.stdout.strip().split(", ")]
            return albums
        return []
    except Exception:
        return []


def import_to_photos_with_albums(
    photo_paths: list[str],
    main_album: str = "PhotoPipe",
    additional_album: str | None = None,
    create_additional: bool = False
) -> tuple[bool, str]:
    """
    Import photos to Photos app with album assignment.

    Args:
        photo_paths: List of file paths to import
        main_album: Main album name (always PhotoPipe)
        additional_album: Optional additional album name
        create_additional: Whether to create the additional album if it doesn't exist

    Returns:
        Tuple of (success, message)
    """
    # Build the AppleScript
    # First, import all photos and get their IDs
    photo_list = ", ".join([f'POSIX file "{p}"' for p in photo_paths])

    script = f'''
    tell application "Photos"
        activate

        -- Import photos
        set importedItems to import {{{photo_list}}}

        -- Ensure PhotoPipe album exists
        set photoPipeAlbum to missing value
        try
            set photoPipeAlbum to album "{main_album}"
        on error
            set photoPipeAlbum to make new album named "{main_album}"
        end try

        -- Add to PhotoPipe album
        add importedItems to photoPipeAlbum

'''

    if additional_album:
        if create_additional:
            script += f'''
        -- Create additional album if needed
        set additionalAlbum to missing value
        try
            set additionalAlbum to album "{additional_album}"
        on error
            set additionalAlbum to make new album named "{additional_album}"
        end try

        -- Add to additional album
        add importedItems to additionalAlbum
'''
        else:
            script += f'''
        -- Add to existing additional album
        try
            set additionalAlbum to album "{additional_album}"
            add importedItems to additionalAlbum
        end try
'''

    script += '''
        return count of importedItems
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=120  # Allow more time for large imports
        )
        if result.returncode == 0:
            count = result.stdout.strip()
            albums_msg = f"'{main_album}'"
            if additional_album:
                albums_msg += f" and '{additional_album}'"
            return True, f"Added {count} photos to {albums_msg}"
        else:
            return False, f"Failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Import timed out - try with fewer photos"
    except Exception as e:
        return False, f"Error: {str(e)}"


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

    if "just_finalized_batch_id" not in st.session_state:
        st.session_state.just_finalized_batch_id = None

    if "finalize_report" not in st.session_state:
        st.session_state.finalize_report = None

    if "photos_import_pending" not in st.session_state:
        st.session_state.photos_import_pending = None  # batch_id if pending import

    if "available_albums" not in st.session_state:
        st.session_state.available_albums = None


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

    # Calculate output folder
    output_folder = generate_output_folder(batch)

    st.write("**Output Location:**")
    st.code(str(output_folder))

    # Show folder structure - fronts only
    batch_name_safe = sanitize_filename(batch.name)
    date_str = batch.date_start.strftime('%Y-%m-%d') if batch.date_start else 'YYYY-MM-DD'

    st.write("**Output Files (fronts only, with embedded metadata):**")
    structure = f"""
{config.paths.output_folder}/
├── {output_folder.relative_to(config.paths.output_folder)}/
│   ├── PP_{date_str}_{batch_name_safe}_0001.jpg
│   ├── PP_{date_str}_{batch_name_safe}_0002.jpg
│   ├── PP_{date_str}_{batch_name_safe}_0003.jpg
│   ├── ...
│   └── _batch_report.json
└── _archive/
    └── {batch_name_safe}/
        └── (original fronts + backs preserved)
    """
    st.code(structure)

    st.caption("PP_ prefix ensures filenames won't conflict with iPhone or camera photos.")

    # Archive settings
    st.write("**Settings:**")
    preserve = st.checkbox(
        "Keep original files in archive (including backs)",
        value=config.output.preserve_originals,
    )

    return preserve


def show_photos_import_ui(batch: Batch, photos_in_folder: list[Path], key_suffix: str = ""):
    """Show Photos app import UI with album selection."""
    st.markdown("---")
    st.subheader("📱 Add to Photos App")

    if not photos_in_folder:
        st.warning("No photos found to import")
        return

    st.write(f"**{len(photos_in_folder)} photos** will be added to the **PhotoPipe** album.")

    # Get available albums
    if st.session_state.available_albums is None:
        with st.spinner("Loading albums from Photos..."):
            st.session_state.available_albums = get_photos_albums()

    albums = st.session_state.available_albums or []

    # Album selection
    st.write("**Additional Album (optional):**")

    album_choice = st.radio(
        "Add to additional album?",
        options=["none", "existing", "new"],
        format_func=lambda x: {
            "none": "PhotoPipe album only",
            "existing": "Also add to existing album",
            "new": "Create new album"
        }[x],
        horizontal=True,
        key=f"album_choice_{key_suffix}"
    )

    additional_album = None
    create_new = False

    if album_choice == "existing" and albums:
        # Filter out PhotoPipe from the list
        other_albums = [a for a in albums if a != "PhotoPipe"]
        if other_albums:
            additional_album = st.selectbox(
                "Select album",
                options=other_albums,
                key=f"select_album_{key_suffix}"
            )
        else:
            st.info("No other albums found. Create a new one instead.")
            album_choice = "new"

    if album_choice == "new":
        additional_album = st.text_input(
            "New album name",
            value=batch.name,
            key=f"new_album_{key_suffix}"
        )
        create_new = True

    # Import button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Import to Photos", type="primary", use_container_width=True, key=f"do_import_{key_suffix}"):
            photo_paths = [str(p) for p in photos_in_folder]

            with st.spinner(f"Importing {len(photo_paths)} photos to Photos app..."):
                success, message = import_to_photos_with_albums(
                    photo_paths,
                    main_album="PhotoPipe",
                    additional_album=additional_album if album_choice != "none" else None,
                    create_additional=create_new
                )

            if success:
                st.success(message)
                # Refresh albums list
                st.session_state.available_albums = None
            else:
                st.error(message)

    with col2:
        if st.button("Skip", use_container_width=True, key=f"skip_import_{key_suffix}"):
            st.session_state.photos_import_pending = None
            st.rerun()


def show_finalize_success(batch: Batch, report):
    """Show success UI after finalization."""
    output_folder = generate_output_folder(batch)
    photos_in_folder = list(output_folder.glob("*.jpg")) + list(output_folder.glob("*.jpeg"))

    st.success(f"""
    ✅ **Batch Finalized Successfully!**

    - Photos processed: {report.photo_count}
    - Photos in output folder: {len(photos_in_folder)}
    - Output folder: `{output_folder}`
    """)

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("📂 Open Output Folder", type="primary", use_container_width=True, key="open_folder_success"):
            subprocess.run(["open", str(output_folder)])

    with btn_col2:
        show_import = st.session_state.photos_import_pending == batch.id
        if st.button(
            "📱 Add to Photos App" if not show_import else "Hide Import Options",
            use_container_width=True,
            key="add_photos_success"
        ):
            if show_import:
                st.session_state.photos_import_pending = None
            else:
                st.session_state.photos_import_pending = batch.id
                st.session_state.available_albums = None  # Refresh albums
            st.rerun()

    with btn_col3:
        if st.button("🔄 Start New Batch", use_container_width=True, key="new_batch_success"):
            st.session_state.just_finalized_batch_id = None
            st.session_state.finalize_report = None
            st.session_state.photos_import_pending = None
            st.switch_page("pages/1_batch_setup.py")

    # Show import UI if pending
    if st.session_state.photos_import_pending == batch.id:
        show_photos_import_ui(batch, photos_in_folder, key_suffix="success")

    # Show report summary
    with st.expander("📋 View Batch Report"):
        st.json({
            "batch_name": report.batch_name,
            "photo_count": report.photo_count,
            "date_range": report.date_range,
            "date_source_breakdown": report.date_source_breakdown,
            "people_tagged": report.people_tagged,
        })


def finalize_controls(batch: Batch, stats: dict):
    """Render finalize controls."""
    st.subheader("🚀 Finalize Batch")

    db = st.session_state.db

    # Check if we just finalized this batch
    if st.session_state.just_finalized_batch_id == batch.id and st.session_state.finalize_report:
        show_finalize_success(batch, st.session_state.finalize_report)
        return

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
            "Auto-approve AI dates",
            value=True,
            help="Automatically approve photos with AI-estimated or high-confidence dates",
        )

    with col2:
        finalize_all = st.checkbox(
            "Finalize all photos",
            value=True,
            help="Finalize all photos even if not individually reviewed (recommended)",
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
                    finalize_all=finalize_all,
                    progress_callback=update_progress,
                )

            # Store in session state
            st.session_state.just_finalized_batch_id = batch.id
            st.session_state.finalize_report = report

            # Update batch status
            batch.status = "complete"
            db.update_batch(batch)

            st.session_state.finalizing = False
            st.rerun()

        except Exception as e:
            st.error(f"❌ Finalization failed: {e}")
            st.session_state.finalizing = False


def completed_batches():
    """Show list of completed batches."""
    st.subheader("✅ Completed Batches")

    db = st.session_state.db
    batches = db.get_all_batches()
    # Skip the just-finalized batch (it's shown in the success UI above)
    completed = [b for b in batches if b.status == "complete" and b.id != st.session_state.just_finalized_batch_id]

    if not completed:
        st.info("No other completed batches.")
        return

    for batch in completed:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.write(f"**{batch.name}**")
            st.caption(batch.get_date_range_str())

        with col2:
            stats = db.get_batch_stats(batch.id)
            st.write(f"📷 {stats['total']}")

        output_folder = generate_output_folder(batch)
        photos_in_folder = list(output_folder.glob("*.jpg")) + list(output_folder.glob("*.jpeg")) if output_folder.exists() else []

        with col3:
            if output_folder.exists():
                if st.button(f"📂 Open ({len(photos_in_folder)})", key=f"open_{batch.id}", use_container_width=True):
                    subprocess.run(["open", str(output_folder)])
            else:
                st.caption("No folder")

        with col4:
            if output_folder.exists() and photos_in_folder:
                show_import = st.session_state.photos_import_pending == batch.id
                if st.button(
                    "📱 Photos" if not show_import else "Hide",
                    key=f"photos_{batch.id}",
                    use_container_width=True
                ):
                    if show_import:
                        st.session_state.photos_import_pending = None
                    else:
                        st.session_state.photos_import_pending = batch.id
                        st.session_state.available_albums = None
                    st.rerun()

        # Show import UI if this batch is selected
        if st.session_state.photos_import_pending == batch.id:
            show_photos_import_ui(batch, photos_in_folder, key_suffix=f"completed_{batch.id}")

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
    preserve = output_preview(batch)

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
        if st.button("← Back to Curate"):
            st.switch_page("pages/3_curate.py")
    with col2:
        if st.button("Start New Batch →"):
            st.switch_page("pages/1_batch_setup.py")


if __name__ == "__main__":
    main()
