"""Helper Mode: minimal UI for someone other than the owner to scan.

Designed for a non-owner (e.g., a family member) to scan stacks of photos
into a free-text "bucket" without needing to know about batches, metadata,
or AI dating. The owner curates the buckets into real batches later.
"""

import streamlit as st

from photopipe.bucket_service import BucketService
from photopipe.capture_pipeline import capture_batch, CaptureProgress
from photopipe.config import get_config
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail

st.set_page_config(
    page_title="Scan Photos",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide sidebar nav for helper mode — keep the UI as bare as possible.
st.markdown(
    """
<style>
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"] {display: none;}
    section.main {padding-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialise the minimum session state this page needs."""
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    if "current_bucket_id" not in st.session_state:
        st.session_state.current_bucket_id = None
    if "helper_name" not in st.session_state:
        st.session_state.helper_name = ""


def _bucket_selection_screen(svc: BucketService) -> None:
    """First screen: pick a label, optionally enter helper name, open bucket."""
    st.markdown("### Where are these photos from?")
    label = st.text_input(
        "Label",
        placeholder="e.g., Grandma's blue album, page 3",
        label_visibility="collapsed",
    )
    helper = st.text_input(
        "Your name (optional)",
        value=st.session_state.helper_name,
    )
    if st.button(
        "🟢 Start scanning",
        type="primary",
        use_container_width=True,
        disabled=not label.strip(),
    ):
        st.session_state.helper_name = helper
        bucket = svc.open_bucket(
            label=label.strip(), helper_name=helper or None
        )
        st.session_state.current_bucket_id = bucket.id
        st.rerun()


def _scan_and_progress(bucket) -> None:
    """Scan button + progress handler. Reruns the page on completion."""
    if not st.button("🟢 Scan Stack", type="primary", use_container_width=True):
        return

    progress_box = st.empty()

    def on_progress(p: CaptureProgress) -> None:
        msg = p.message or p.stage
        if p.total:
            progress_box.progress(
                p.current / p.total, text=f"{msg} ({p.current}/{p.total})"
            )
        else:
            progress_box.info(msg)

    with st.spinner("Scanning..."):
        cfg = get_config()
        result = capture_batch(
            bucket,
            db=st.session_state.db,
            scanner_device=cfg.scanner.device or "epsonds:net:192.168.1.62",
            resolution=cfg.scanner.resolution,
            duplex=cfg.scanner.duplex,
            progress=on_progress,
        )

    if result.errors:
        for err in result.errors:
            st.error(err)
    if result.photos_added:
        st.success(f"✓ Added {result.photos_added} photos to this bucket")
    st.rerun()


def _thumbnail_grid(photos) -> None:
    """Show the last 18 captured photos as a 6-wide thumbnail grid."""
    if not photos:
        return
    st.markdown(f"**{len(photos)} photos in this bucket**")
    cols = st.columns(6)
    for i, photo in enumerate(photos[-18:]):
        with cols[i % 6]:
            try:
                st.image(
                    generate_thumbnail(photo.front_path, size=(120, 120)),
                    use_container_width=True,
                )
            except Exception:
                st.caption(f"#{photo.sequence_num}")


def main() -> None:
    init_state()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📷 Scan Photos")

    if st.session_state.current_bucket_id is None:
        _bucket_selection_screen(svc)
        return

    bucket = db.get_bucket(st.session_state.current_bucket_id)
    if bucket is None:
        # Stale session reference — start over.
        st.session_state.current_bucket_id = None
        st.rerun()
        return

    st.subheader(f"📁 {bucket.label}")
    if bucket.helper_name:
        st.caption(f"Scanned by {bucket.helper_name}")

    _scan_and_progress(bucket)

    photos = db.get_photos_by_bucket(bucket.id)
    _thumbnail_grid(photos)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Done with this bucket", use_container_width=True):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()
    with col2:
        if st.button("➕ Start a new bucket", use_container_width=True):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()


if __name__ == "__main__":
    main()
