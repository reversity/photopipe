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
    if "scan_errors" not in st.session_state:
        st.session_state.scan_errors = []


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
        help="Records who scanned this stack, so the owner knows who to ask about it later.",
    )
    if st.button(
        "🟢 Start scanning",
        type="primary",
        use_container_width=True,
        disabled=not label.strip(),
        help="Opens a new bucket with this label. You can scan several stacks into the same bucket before pressing Done.",
    ):
        st.session_state.helper_name = helper
        bucket = svc.open_bucket(
            label=label.strip(), helper_name=helper or None
        )
        st.session_state.current_bucket_id = bucket.id
        st.rerun()


def _scan_and_progress(bucket) -> None:
    """Scan button + progress handler. Reruns the page on completion."""
    if not st.button("🟢 Scan Stack", type="primary", use_container_width=True, help="Runs the scanner and adds everything in the feeder to this bucket. Load the next stack and press again to keep going."):
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
            scanner_device=cfg.scanner.device,
            resolution=cfg.scanner.resolution,
            duplex=cfg.scanner.duplex,
            progress=on_progress,
        )

    if result.errors:
        st.session_state.scan_errors = list(result.errors)
    if result.photos_added:
        st.success(f"✓ Added {result.photos_added} photos to this bucket")
    st.rerun()


def _show_scan_errors() -> None:
    """Show errors from the last scan, persisted across the post-scan rerun."""
    if not st.session_state.scan_errors:
        return
    for err in st.session_state.scan_errors:
        st.error(err)
    if st.button("Dismiss errors"):
        st.session_state.scan_errors = []
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


def _owner_exit_footer() -> None:
    """Discreet way back to owner mode.

    Helper Mode hides the sidebar entirely, so without this the owner
    would have to restart the app or hand-edit the URL to get back.
    Kept in a collapsed expander so a helper is unlikely to wander in.
    """
    st.markdown("---")
    with st.expander("⚙️ Owner"):
        st.caption("Leave Helper Mode and return to the full owner view.")
        if st.button("Exit Helper Mode"):
            st.session_state.helper_mode = False
            st.switch_page("app.py")


def main() -> None:
    init_state()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📷 Scan Photos")
    st.caption("Helper Mode — the first step of the workflow: scan stacks of photos into a labeled bucket. The owner turns buckets into batches later.")

    if st.session_state.current_bucket_id is None:
        _bucket_selection_screen(svc)
        _owner_exit_footer()
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
    _show_scan_errors()

    photos = db.get_photos_by_bucket(bucket.id)
    _thumbnail_grid(photos)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Done with this bucket", use_container_width=True, help="Closes the bucket so the owner can turn it into a batch. No more photos can be added to it after this."):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()
    with col2:
        if st.button("➕ Start a new bucket", use_container_width=True, help="Closes this bucket and returns to the label screen so you can begin a new one — use this when you switch albums or boxes."):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()

    _owner_exit_footer()


if __name__ == "__main__":
    main()
