"""Owner-facing bucket dashboard: list, view stats, convert to batch."""
import streamlit as st
from photopipe.bucket_service import BucketService
from photopipe.config import get_config
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import BucketStatus

st.set_page_config(page_title="Buckets - PhotoPipe", page_icon="📦", layout="wide")


def main():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📦 Buckets")
    st.caption("Raw scans waiting to be curated into batches.")

    show_all = st.checkbox("Show converted buckets", value=False)
    # Note: Bucket has use_enum_values=True, so bucket.status is a plain string
    # (e.g. "open", "closed", "converted"). Comparisons against BucketStatus.X
    # still work because BucketStatus is a str-Enum mixin.
    buckets = svc.db.list_buckets() if show_all else [
        b for b in svc.db.list_buckets() if b.status != BucketStatus.CONVERTED
    ]

    if not buckets:
        st.info("No open buckets. Switch to Helper Mode and scan some photos.")
        return

    for bucket in buckets:
        stats = svc.get_stats(bucket.id)
        with st.expander(
            f"📁 {bucket.label}  ·  {stats.photo_count} photos  "
            f"·  {'helper: ' + stats.helper_name if stats.helper_name else 'unattributed'}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.metric("Photos", stats.photo_count)
            with col2:
                st.metric("With dates from back", stats.photos_with_extracted_date)
            with col3:
                # bucket.status is already a string thanks to use_enum_values
                st.write(f"**Status:** {bucket.status}")

            # Thumbnail strip (first 6)
            photos = db.get_photos_by_bucket(bucket.id)
            if photos:
                cols = st.columns(6)
                for i, p in enumerate(photos[:6]):
                    with cols[i]:
                        try:
                            st.image(generate_thumbnail(p.front_path, size=(120, 120)))
                        except Exception:
                            st.caption(f"#{p.sequence_num}")

            if bucket.status == BucketStatus.CLOSED:
                with st.form(f"convert_{bucket.id}"):
                    st.markdown("### Convert to Batch")
                    name = st.text_input("Batch name", value=bucket.label)
                    c1, c2 = st.columns(2)
                    with c1:
                        date_start = st.date_input("Date start (optional)", value=None)
                    with c2:
                        date_end = st.date_input("Date end (optional)", value=None)
                    location = st.text_input("Location (optional)")
                    people = st.text_input("People (comma-separated, optional)")
                    event = st.text_area("Event description (optional)", height=60)
                    if st.form_submit_button("✅ Convert to Batch", type="primary"):
                        batch = svc.convert_to_batch(
                            bucket.id,
                            name=name,
                            date_start=date_start or None,
                            date_end=date_end or None,
                            location_description=location or None,
                            people=[p.strip() for p in people.split(",") if p.strip()],
                            event_description=event or None,
                        )
                        st.success(f"Converted to batch '{batch.name}'")
                        st.rerun()

            if bucket.status == BucketStatus.CONVERTED:
                st.info(f"Already converted to batch (id: {bucket.batch_id})")


if __name__ == "__main__":
    main()
