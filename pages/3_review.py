"""
Review Queue Page - Review and approve photo metadata.
"""

import streamlit as st
from datetime import date
from pathlib import Path
import io

from PIL import Image

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import (
    Batch,
    PhotoPair,
    PhotoStatus,
    DateSource,
    DateConfidence,
    Location,
)
from photopipe.geocoding import geocode_location
from photopipe.file_manager import generate_thumbnail


st.set_page_config(page_title="Review Queue - PhotoPipe", page_icon="👁️", layout="wide")


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None

    if "current_photo_index" not in st.session_state:
        st.session_state.current_photo_index = 0

    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "all"


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
        "Select Batch",
        options=list(batch_options.keys()),
    )

    st.session_state.current_batch_id = batch_options[selected_name]
    return db.get_batch(st.session_state.current_batch_id)


def filter_controls(batch: Batch):
    """Render filter controls."""
    db = st.session_state.db

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        filter_mode = st.selectbox(
            "Show",
            options=["all", "needs_review", "low_confidence", "ai_estimated", "unreviewed"],
            format_func=lambda x: {
                "all": "All Photos",
                "needs_review": "Needs Review",
                "low_confidence": "Low Confidence",
                "ai_estimated": "AI Estimated",
                "unreviewed": "Not Yet Reviewed",
            }[x],
        )
        st.session_state.filter_mode = filter_mode

    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["sequence", "date_asc", "date_desc", "confidence"],
            format_func=lambda x: {
                "sequence": "Sequence #",
                "date_asc": "Date (Oldest First)",
                "date_desc": "Date (Newest First)",
                "confidence": "Confidence",
            }[x],
        )

    # Get stats
    stats = db.get_batch_stats(batch.id)

    with col3:
        st.metric("Total", stats["total"])

    with col4:
        st.metric("Needs Review", stats["needs_review"])

    return filter_mode, sort_by


def get_filtered_photos(batch_id: str, filter_mode: str, sort_by: str) -> list[PhotoPair]:
    """Get filtered and sorted photos."""
    db = st.session_state.db
    photos = db.get_photos_by_batch(batch_id)

    # Apply filter
    if filter_mode == "needs_review":
        photos = [p for p in photos if p.needs_review]
    elif filter_mode == "low_confidence":
        photos = [p for p in photos if p.date_confidence == DateConfidence.LOW]
    elif filter_mode == "ai_estimated":
        photos = [p for p in photos if p.date_source == DateSource.AI_ESTIMATED]
    elif filter_mode == "unreviewed":
        photos = [p for p in photos if p.status == PhotoStatus.PROCESSED]

    # Apply sort
    if sort_by == "date_asc":
        photos.sort(key=lambda p: p.extracted_date or date.max)
    elif sort_by == "date_desc":
        photos.sort(key=lambda p: p.extracted_date or date.min, reverse=True)
    elif sort_by == "confidence":
        confidence_order = {"high": 0, "medium": 1, "low": 2, None: 3}
        photos.sort(key=lambda p: confidence_order.get(p.date_confidence.value if p.date_confidence else None, 3))
    # Default is sequence order

    return photos


def display_photo_image(photo: PhotoPair, side: str):
    """Display photo image with proper handling."""
    path = photo.front_path if side == "front" else photo.back_path

    if not path or not path.exists():
        st.info(f"No {side} image")
        return

    try:
        img = Image.open(path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load image: {e}")


def photo_review_card(photo: PhotoPair, batch: Batch, index: int):
    """Render a single photo review card."""
    db = st.session_state.db

    with st.container():
        # Header
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.subheader(f"Photo #{photo.sequence_num}")

        with col2:
            status_colors = {
                "ingested": "🔵",
                "processed": "🟡",
                "reviewed": "🟢",
                "finalized": "✅",
            }
            st.write(f"Status: {status_colors.get(photo.status, '⚪')} {photo.status.title()}")

        with col3:
            if photo.needs_review:
                st.warning("⚠️ Needs Review")

        # Three-column layout
        col_front, col_back, col_meta = st.columns([2, 2, 3])

        with col_front:
            st.write("**Front**")
            display_photo_image(photo, "front")

        with col_back:
            st.write("**Back**")
            if photo.back_path:
                display_photo_image(photo, "back")

                # Show OCR text
                if photo.ocr_text_back:
                    with st.expander("OCR Text"):
                        st.text(photo.ocr_text_back)
            else:
                st.info("No back scan")

        with col_meta:
            st.write("**Metadata**")

            # Date section
            confidence_icons = {
                "high": "🟢",
                "medium": "🟡",
                "low": "🔴",
            }

            extracted_date = photo.extracted_date
            if extracted_date:
                conf_icon = confidence_icons.get(photo.date_confidence.value if photo.date_confidence else "low", "⚪")
                st.write(f"Extracted Date: {extracted_date} {conf_icon}")
                st.caption(f"Source: {photo.date_source.value if photo.date_source else 'Unknown'}")
            else:
                st.write("No date extracted")

            # Editable final date
            with st.form(key=f"edit_photo_{photo.id}"):
                final_date = st.date_input(
                    "Final Date",
                    value=photo.final_date or photo.extracted_date,
                )

                # Use batch location if no photo-specific location
                location_value = ""
                if photo.final_location:
                    location_value = photo.final_location.description
                elif batch.location_description:
                    location_value = batch.location_description

                final_location = st.text_input(
                    "Location",
                    value=location_value,
                )

                final_description = st.text_area(
                    "Description",
                    value=photo.final_description or batch.event_description or "",
                    height=80,
                )

                # Keywords from batch people
                default_keywords = ", ".join(photo.final_keywords) if photo.final_keywords else ", ".join(batch.people)
                keywords_input = st.text_input(
                    "Keywords/People",
                    value=default_keywords,
                )

                review_notes = st.text_input(
                    "Review Notes",
                    value=photo.review_notes or "",
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    approve = st.form_submit_button("✅ Approve", type="primary")
                with col2:
                    save_draft = st.form_submit_button("💾 Save Draft")
                with col3:
                    flag = st.form_submit_button("🚩 Flag")

                if approve or save_draft or flag:
                    # Update photo
                    photo.final_date = final_date

                    # Geocode location if changed
                    if final_location and final_location != location_value:
                        loc = geocode_location(final_location)
                        if loc:
                            photo.final_location = loc
                    elif final_location and batch.location:
                        photo.final_location = batch.location

                    photo.final_description = final_description
                    photo.final_keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
                    photo.review_notes = review_notes

                    if approve:
                        photo.status = PhotoStatus.REVIEWED
                        photo.needs_review = False
                        st.success("Photo approved!")
                    elif flag:
                        photo.needs_review = True
                        photo.status = PhotoStatus.PROCESSED
                        st.warning("Photo flagged for review")
                    else:
                        st.info("Draft saved")

                    db.update_photo(photo)
                    db.log_action(
                        photo_id=photo.id,
                        batch_id=batch.id,
                        action="reviewed" if approve else "draft_saved",
                        details={"final_date": str(final_date)},
                    )
                    st.rerun()

            # Show AI analysis if available
            if photo.ai_analysis:
                with st.expander("🤖 AI Analysis"):
                    ai = photo.get_ai_estimate()
                    if ai:
                        st.write(f"**Estimated Year:** {ai.year}")
                        st.write(f"**Confidence:** {ai.confidence.value}")
                        st.write("**Evidence:**")
                        for ev in ai.evidence:
                            st.write(f"- {ev}")
                        st.write(f"**Reasoning:** {ai.reasoning}")

        st.markdown("---")


def bulk_actions(batch: Batch, photos: list[PhotoPair]):
    """Render bulk action controls."""
    st.subheader("Bulk Actions")

    db = st.session_state.db

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve All High Confidence"):
            approved = 0
            for photo in photos:
                if photo.date_confidence == DateConfidence.HIGH and photo.extracted_date:
                    photo.final_date = photo.extracted_date
                    photo.final_location = batch.location
                    photo.status = PhotoStatus.REVIEWED
                    photo.needs_review = False
                    db.update_photo(photo)
                    approved += 1
            st.success(f"Approved {approved} photos")
            st.rerun()

    with col2:
        if st.button("📅 Apply Batch Date to Undated"):
            updated = 0
            total = len(photos)
            for i, photo in enumerate(photos):
                if not photo.extracted_date and not photo.final_date:
                    photo.final_date = batch.calculate_date_for_sequence(i + 1, total)
                    photo.date_source = DateSource.BATCH_DEFAULT
                    db.update_photo(photo)
                    updated += 1
            st.success(f"Applied batch date to {updated} photos")
            st.rerun()

    with col3:
        if st.button("📍 Apply Batch Location to All"):
            if batch.location:
                for photo in photos:
                    if not photo.final_location:
                        photo.final_location = batch.location
                        db.update_photo(photo)
                st.success(f"Applied location to {len(photos)} photos")
                st.rerun()
            else:
                st.warning("Batch has no location set")


def thumbnail_grid(photos: list[PhotoPair], batch: Batch):
    """Display thumbnail grid for quick overview."""
    st.subheader("Quick Overview")

    if not photos:
        st.info("No photos to display")
        return

    # Display in grid
    cols_per_row = 5
    rows = (len(photos) + cols_per_row - 1) // cols_per_row

    for row in range(min(rows, 4)):  # Limit to 4 rows
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            photo_idx = row * cols_per_row + col_idx
            if photo_idx < len(photos):
                photo = photos[photo_idx]
                with cols[col_idx]:
                    try:
                        thumb = generate_thumbnail(photo.front_path, (150, 150))
                        st.image(thumb, use_container_width=True)

                        # Status indicator
                        if photo.needs_review:
                            st.caption(f"#{photo.sequence_num} ⚠️")
                        elif photo.status == PhotoStatus.REVIEWED:
                            st.caption(f"#{photo.sequence_num} ✅")
                        else:
                            st.caption(f"#{photo.sequence_num}")
                    except Exception:
                        st.write(f"#{photo.sequence_num}")

    if len(photos) > rows * cols_per_row:
        st.caption(f"Showing {rows * cols_per_row} of {len(photos)} photos")


def main():
    """Main page."""
    init_session_state()

    st.title("👁️ Review Queue")
    st.write("Review and approve metadata for your photos.")

    # Batch selector
    batch = batch_selector()
    if not batch:
        return

    st.markdown("---")

    # Filter controls
    filter_mode, sort_by = filter_controls(batch)

    # Get filtered photos
    photos = get_filtered_photos(batch.id, filter_mode, sort_by)

    if not photos:
        st.info(f"No photos match the filter '{filter_mode}'")
        return

    st.write(f"Showing **{len(photos)}** photos")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 List View", "🖼️ Grid View", "⚡ Bulk Actions"])

    with tab1:
        # Pagination
        photos_per_page = get_config().ui.thumbnails_per_page
        total_pages = (len(photos) + photos_per_page - 1) // photos_per_page

        if total_pages > 1:
            page = st.slider("Page", 1, total_pages, 1)
        else:
            page = 1

        start_idx = (page - 1) * photos_per_page
        end_idx = min(start_idx + photos_per_page, len(photos))

        for i, photo in enumerate(photos[start_idx:end_idx]):
            photo_review_card(photo, batch, start_idx + i)

    with tab2:
        thumbnail_grid(photos, batch)

    with tab3:
        bulk_actions(batch, photos)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Scan Ingest"):
            st.switch_page("pages/2_scan_ingest.py")
    with col2:
        if st.button("Continue to Finalize →"):
            st.switch_page("pages/4_finalize.py")


if __name__ == "__main__":
    main()
