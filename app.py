"""
PhotoPipe - Photo Scanning Metadata Pipeline

Main Streamlit application entry point.
"""

import streamlit as st
from pathlib import Path

from photopipe.config import get_config, reload_config
from photopipe.database import Database
from photopipe.setup import is_setup_complete, load_settings


# Page configuration
st.set_page_config(
    page_title="PhotoPipe",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()

    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None

    if "watcher_running" not in st.session_state:
        st.session_state.watcher_running = False

    if "helper_mode" not in st.session_state:
        st.session_state.helper_mode = False


def main():
    """Main application."""
    # Check if first-run setup is needed
    if not is_setup_complete():
        st.warning("Welcome to PhotoPipe! Please complete the initial setup.")
        if st.button("Go to Setup", type="primary"):
            st.switch_page("pages/0_setup.py")
        st.stop()

    init_session_state()

    # One-shot scanner-discovery check (runs once per session). On macOS
    # Tahoe (26+) the system silently drops Local Network permission after
    # updates, which makes networked scanners disappear without warning.
    if "scanner_check_done" not in st.session_state:
        st.session_state.scanner_check_done = True
        from photopipe.cli.doctor import check_scanner_discovery
        check = check_scanner_discovery()
        if not check.ok:
            st.session_state.scanner_warning = check.fix or check.detail

    if st.session_state.get("scanner_warning"):
        st.warning(f"⚠️ Scanner not detected. {st.session_state.scanner_warning}")

    # Helper mode short-circuits the home page entirely: a non-owner helper
    # who lands on the app just gets the bare scan screen.
    if st.session_state.get("helper_mode"):
        st.switch_page("pages/2_capture.py")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=PhotoPipe", use_container_width=True)
        st.markdown("---")

        # Quick stats
        db = st.session_state.db
        batches = db.get_all_batches()

        st.metric("Total Batches", len(batches))

        pending = len([b for b in batches if b.status == "pending"])
        review = len([b for b in batches if b.status == "review"])
        complete = len([b for b in batches if b.status == "complete"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pending", pending)
        with col2:
            st.metric("Complete", complete)

        if review > 0:
            st.warning(f"⚠️ {review} batch(es) need review")

        st.markdown("---")

        # Current batch selector
        if batches:
            batch_options = {b.name: b.id for b in batches}
            selected_name = st.selectbox(
                "Current Batch",
                options=list(batch_options.keys()),
                index=0 if st.session_state.current_batch_id is None else None,
            )
            if selected_name:
                st.session_state.current_batch_id = batch_options[selected_name]

        st.markdown("---")

        # System status
        st.subheader("System Status")

        config = get_config()

        # Check dependencies
        from photopipe.metadata import check_exiftool_installed
        exiftool_ok = check_exiftool_installed()

        import shutil
        tesseract_ok = shutil.which("tesseract") is not None

        from photopipe.ai_dating import is_ai_dating_available
        ai_ok = is_ai_dating_available()

        st.write("ExifTool:", "✅" if exiftool_ok else "❌")
        st.write("Tesseract:", "✅" if tesseract_ok else "❌")
        st.write("AI Dating:", "✅" if ai_ok else "⚠️ No API key")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚙️ Settings", use_container_width=True):
                st.switch_page("pages/0_setup.py")
        with col2:
            if st.button("⚙️ Config", use_container_width=True):
                st.switch_page("pages/settings.py")

        st.markdown("---")
        # Helper Mode: hides owner-only pages and redirects to the bare
        # scan UI. Useful when handing the scanner to a family member.
        helper_mode = st.toggle(
            "Helper Mode",
            value=st.session_state.get("helper_mode", False),
            help="Hide owner-only pages and show only the scan screen.",
        )
        if helper_mode != st.session_state.get("helper_mode", False):
            st.session_state.helper_mode = helper_mode
            st.rerun()

    # Main content
    st.markdown('<p class="main-header">📷 PhotoPipe</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Photo Scanning Metadata Pipeline for Epson FastFoto</p>',
        unsafe_allow_html=True,
    )

    # Welcome message and quick actions
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📁 Batch Setup")
        st.write("Create and manage photo batches")
        if st.button("Go to Batch Setup", key="goto_batch"):
            st.switch_page("pages/1_batch_setup.py")

    with col2:
        st.markdown("### 📥 Capture")
        st.write("Scan photos into a bucket")
        if st.button("Go to Capture", key="goto_capture"):
            st.switch_page("pages/2_capture.py")

    with col3:
        st.markdown("### 👁️ Curate")
        st.write("Review buckets and run AI dating")
        if st.button("Go to Curate", key="goto_curate"):
            st.switch_page("pages/3_curate.py")

    with col4:
        st.markdown("### ✅ Finalize")
        st.write("Export with embedded metadata")
        if st.button("Go to Finalize", key="goto_finalize"):
            st.switch_page("pages/4_finalize.py")

    # Recent activity
    st.markdown("---")
    st.subheader("Recent Activity")

    db = st.session_state.db
    logs = db.get_logs(limit=10)

    if logs:
        for log in logs:
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.write(log.timestamp.strftime("%Y-%m-%d %H:%M"))
            with col2:
                st.write(log.action.replace("_", " ").title())
            with col3:
                if log.batch_id:
                    batch = db.get_batch(log.batch_id)
                    if batch:
                        st.write(f"Batch: {batch.name}")
    else:
        st.info("No recent activity. Create a batch to get started!")

    # Footer
    st.markdown("---")
    st.caption("PhotoPipe v0.1.0 | [Settings](pages/settings.py)")


if __name__ == "__main__":
    main()
