"""
Batch Setup Page - Create and manage photo batches.
"""

import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import re

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import Batch, BatchTemplate, BatchStatus
from photopipe.geocoding import geocode_location
from photopipe.setup import load_settings


st.set_page_config(page_title="Batch Setup - PhotoPipe", page_icon="📁", layout="wide")


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "editing_batch_id" not in st.session_state:
        st.session_state.editing_batch_id = None



def parse_approximate_date(text: str) -> tuple[date | None, date | None]:
    """
    Parse approximate date text into a date range.

    Supports:
    - "1985" -> Jan 1 1985 to Dec 31 1985
    - "June 1985" or "Jun 1985" -> June 1-30, 1985
    - "Summer 1985" -> June 1 - Aug 31, 1985
    - "Spring 1985" -> March 1 - May 31, 1985
    - "Fall 1985" or "Autumn 1985" -> Sept 1 - Nov 30, 1985
    - "Winter 1985" -> Dec 1, 1984 - Feb 28, 1985
    - "Early 1985" -> Jan-Apr 1985
    - "Mid 1985" -> May-Aug 1985
    - "Late 1985" -> Sept-Dec 1985
    """
    if not text:
        return None, None

    text = text.strip().lower()

    # Month names
    months = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    # Days in each month (non-leap year default)
    month_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    # Extract year (required)
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if not year_match:
        return None, None
    year = int(year_match.group(1))

    # Check for leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    if is_leap:
        month_days[2] = 29

    # Check for season
    if 'winter' in text:
        # Winter spans year boundary
        return date(year - 1, 12, 1), date(year, 2, month_days[2])
    elif 'spring' in text:
        return date(year, 3, 1), date(year, 5, 31)
    elif 'summer' in text:
        return date(year, 6, 1), date(year, 8, 31)
    elif 'fall' in text or 'autumn' in text:
        return date(year, 9, 1), date(year, 11, 30)

    # Check for early/mid/late
    if 'early' in text:
        return date(year, 1, 1), date(year, 4, 30)
    elif 'mid' in text:
        return date(year, 5, 1), date(year, 8, 31)
    elif 'late' in text:
        return date(year, 9, 1), date(year, 12, 31)

    # Check for month
    for month_name, month_num in months.items():
        if month_name in text:
            return date(year, month_num, 1), date(year, month_num, month_days[month_num])

    # Just year
    return date(year, 1, 1), date(year, 12, 31)


def create_batch_form():
    """Render the batch creation form."""
    st.subheader("Create New Batch")

    # Help tip
    with st.expander("💡 What's a batch?", expanded=False):
        st.markdown("""
        A **batch** is a group of related photos - like photos from the same event, vacation, or time period.

        **Examples:**
        - "Christmas 1987" - all photos from that holiday
        - "Summer Vacation 1985" - a trip to the beach
        - "Mom's Birthday 1992" - a single event

        Grouping photos into batches helps you:
        - Apply the same date range and location to all photos
        - Tag all photos with the same people
        - Process and export them together
        """)

    config = get_config()
    db = st.session_state.db

    # Load templates for dropdown
    templates = db.get_all_templates()
    template_options = {"None": None}
    template_options.update({t.name: t for t in templates})

    with st.form("create_batch_form"):
        # Template selector
        selected_template = st.selectbox(
            "Load from Template",
            options=list(template_options.keys()),
        )

        template = template_options.get(selected_template)

        col1, col2 = st.columns(2)

        with col1:
            batch_name = st.text_input(
                "Batch Name *",
                placeholder="e.g., Summer_Vacation_1985",
                help="A descriptive name for this batch of photos",
            )

            # Approximate date - simple text input
            approx_date = st.text_input(
                "Approximate Date",
                placeholder="e.g., Summer 1985, June 1987, 1990",
                help="Enter a year, month+year, or season+year. Examples: '1985', 'June 1985', 'Summer 1985', 'Early 1990s'",
            )

            # Location - use template, then user settings defaults
            user_settings = load_settings()
            default_location = template.location_description if template else (user_settings.default_location or "")
            location_text = st.text_input(
                "Location",
                value=default_location,
                placeholder="e.g., Toledo, OH or 123 Main St, Toledo, OH",
                help="Location will be geocoded to GPS coordinates",
            )

        with col2:
            # Event description
            event_description = st.text_area(
                "Event/Description",
                placeholder="e.g., Summer vacation at Grandma's house. Kids were 5 and 8.",
                height=100,
            )

            # People tags - use template, then user settings defaults
            default_people = template.people if template else user_settings.default_people
            people_input = st.text_input(
                "People in Photos",
                value=", ".join(default_people) if default_people else "",
                placeholder="e.g., Grandma Rose, Mom, Dad, Johnny, Sarah",
                help="Comma-separated list of people likely in these photos",
            )

            # Input folder
            input_folder = st.text_input(
                "Scanner Input Folder",
                value=str(config.paths.input_folder),
                help="Folder where scanner saves images",
            )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            submitted = st.form_submit_button("Create Batch", type="primary")
        with col2:
            save_template = st.form_submit_button("Save as Template")

        if submitted:
            if not batch_name:
                st.error("Batch name is required")
            elif db.get_batch_by_name(batch_name):
                st.error(f"Batch '{batch_name}' already exists")
            else:
                # Parse approximate date
                date_start, date_end = parse_approximate_date(approx_date)
                if approx_date and not date_start:
                    st.warning(f"Could not parse date '{approx_date}'. Use formats like: '1985', 'June 1985', 'Summer 1985'")

                # Parse people
                people = [p.strip() for p in people_input.split(",") if p.strip()]

                # Geocode location
                location = None
                if location_text:
                    with st.spinner("Geocoding location..."):
                        location = geocode_location(location_text)
                    if location:
                        st.success(f"📍 Location: {location.address}")
                    else:
                        st.warning("Could not geocode location. GPS data won't be added.")

                # Create batch
                batch = Batch(
                    name=batch_name,
                    date_start=date_start,
                    date_end=date_end,
                    location_description=location_text,
                    location=location,
                    event_description=event_description,
                    people=people,
                    input_folder=Path(input_folder) if input_folder else None,
                )

                db.create_batch(batch)
                st.success(f"✅ Batch '{batch_name}' created successfully!")
                if date_start and date_end:
                    st.caption(f"Date range: {date_start.strftime('%b %d, %Y')} - {date_end.strftime('%b %d, %Y')}")
                st.session_state.current_batch_id = batch.id
                st.rerun()

        if save_template:
            if not batch_name:
                st.error("Enter a name for the template")
            else:
                people = [p.strip() for p in people_input.split(",") if p.strip()]
                location = geocode_location(location_text) if location_text else None

                template = BatchTemplate(
                    name=batch_name,
                    location_description=location_text,
                    location=location,
                    people=people,
                )
                db.create_template(template)
                st.success(f"✅ Template '{batch_name}' saved!")
                st.rerun()


def batch_list():
    """Render the list of existing batches."""
    st.subheader("Existing Batches")

    db = st.session_state.db
    batches = db.get_all_batches()

    if not batches:
        st.info("No batches yet. Create one above to get started!")
        return

    # Filter options
    col1, col2 = st.columns([3, 1])
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "Pending", "Processing", "Review", "Complete"],
        )

    # Filter batches
    if status_filter != "All":
        batches = [b for b in batches if b.status == status_filter.lower()]

    # Display batches
    for batch in batches:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])

            with col1:
                st.markdown(f"**{batch.name}**")
                st.caption(batch.get_date_range_str())

            with col2:
                if batch.location_description:
                    st.write(f"📍 {batch.location_description}")
                else:
                    st.write("📍 No location")

            with col3:
                stats = db.get_batch_stats(batch.id)
                st.write(f"📷 {stats['total']} photos")
                if stats['needs_review'] > 0:
                    st.caption(f"⚠️ {stats['needs_review']} need review")

            with col4:
                status_colors = {
                    "pending": "🔵",
                    "processing": "🟡",
                    "review": "🟠",
                    "complete": "🟢",
                }
                st.write(f"{status_colors.get(batch.status, '⚪')} {batch.status.title()}")

            with col5:
                if st.button("Edit", key=f"edit_{batch.id}"):
                    st.session_state.editing_batch_id = batch.id
                    st.rerun()
                if st.button("Delete", key=f"delete_{batch.id}"):
                    db.delete_batch(batch.id)
                    st.success(f"Deleted batch '{batch.name}'")
                    st.rerun()

            st.markdown("---")


def edit_batch_form():
    """Render the batch editing form."""
    db = st.session_state.db
    batch = db.get_batch(st.session_state.editing_batch_id)

    if not batch:
        st.error("Batch not found")
        st.session_state.editing_batch_id = None
        return

    st.subheader(f"Edit Batch: {batch.name}")

    with st.form("edit_batch_form"):
        col1, col2 = st.columns(2)

        with col1:
            batch_name = st.text_input("Batch Name *", value=batch.name)

            # Show current date range and allow editing
            current_date_str = batch.get_date_range_str() if batch.date_start else ""
            approx_date = st.text_input(
                "Approximate Date",
                value=current_date_str,
                placeholder="e.g., Summer 1985, June 1987, 1990",
                help="Enter a year, month+year, or season+year",
            )

            location_text = st.text_input(
                "Location",
                value=batch.location_description or "",
            )

        with col2:
            event_description = st.text_area(
                "Event/Description",
                value=batch.event_description or "",
                height=100,
            )

            people_input = st.text_input(
                "People in Photos",
                value=", ".join(batch.people),
            )

            status = st.selectbox(
                "Status",
                options=["pending", "processing", "review", "complete"],
                index=["pending", "processing", "review", "complete"].index(batch.status),
            )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            submitted = st.form_submit_button("Save Changes", type="primary")
        with col2:
            cancel = st.form_submit_button("Cancel")

        if submitted:
            # Parse approximate date
            date_start, date_end = parse_approximate_date(approx_date)

            people = [p.strip() for p in people_input.split(",") if p.strip()]

            # Re-geocode if location changed
            location = batch.location
            if location_text != batch.location_description:
                if location_text:
                    with st.spinner("Geocoding location..."):
                        location = geocode_location(location_text)

            batch.name = batch_name
            batch.date_start = date_start
            batch.date_end = date_end
            batch.location_description = location_text
            batch.location = location
            batch.event_description = event_description
            batch.people = people
            batch.status = BatchStatus(status)

            db.update_batch(batch)
            st.success("Batch updated!")
            st.session_state.editing_batch_id = None
            st.rerun()

        if cancel:
            st.session_state.editing_batch_id = None
            st.rerun()


def template_management():
    """Render template management section."""
    st.subheader("Batch Templates")

    db = st.session_state.db
    templates = db.get_all_templates()

    if not templates:
        st.info("No templates saved yet. Create a batch and click 'Save as Template'.")
        return

    for template in templates:
        col1, col2, col3 = st.columns([3, 3, 1])

        with col1:
            st.write(f"**{template.name}**")

        with col2:
            if template.location_description:
                st.caption(f"📍 {template.location_description}")
            if template.people:
                st.caption(f"👥 {', '.join(template.people[:3])}{'...' if len(template.people) > 3 else ''}")

        with col3:
            if st.button("Delete", key=f"del_template_{template.id}"):
                db.delete_template(template.id)
                st.rerun()


def main():
    """Main page."""
    init_session_state()

    st.title("📁 Batch Setup")
    st.write("Create and manage batches of photos for processing.")

    # Show edit form if editing, otherwise show create form
    if st.session_state.editing_batch_id:
        edit_batch_form()
    else:
        create_batch_form()

    st.markdown("---")

    # Batch list
    batch_list()

    # Templates section in expander
    with st.expander("📋 Manage Templates"):
        template_management()


if __name__ == "__main__":
    main()
