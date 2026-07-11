"""
PhotoPipe - Photo Scanning Metadata Pipeline

Main Streamlit application entry point.
"""

import streamlit as st
from pathlib import Path

from photopipe.config import get_config, reload_config
from photopipe.database import Database
from photopipe.logging_config import setup_logging
from photopipe.setup import is_setup_complete, load_settings

setup_logging()


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

    # An explicit ?mode= in the URL is AUTHORITATIVE on every load, not just
    # the first — so relaunching the PhotoPipe Scanner app (?mode=helper)
    # always returns a helper to the bare scan UI even if someone previously
    # tapped "Exit Helper Mode" in that browser. Session state is otherwise
    # per browser session, so the owner's toggle can't reach other users.
    mode = st.query_params.get("mode", "")
    if mode == "helper":
        st.session_state.helper_mode = True
    elif mode == "owner":
        st.session_state.helper_mode = False
    elif "helper_mode" not in st.session_state:
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
        st.markdown("## 📷 PhotoPipe")
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
            batch_names = list(batch_options.keys())
            batch_ids = list(batch_options.values())
            current_id = st.session_state.current_batch_id
            current_index = batch_ids.index(current_id) if current_id in batch_ids else None
            selected_name = st.selectbox(
                "Current Batch",
                options=batch_names,
                index=current_index,
                placeholder="Select a batch",
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

        from photopipe.vlm_client import is_vlm_available
        ai_ok = is_vlm_available()

        st.write("ExifTool:", "✅" if exiftool_ok else "❌")
        st.write("AI Dating:", "✅" if ai_ok else "⚠️ No API key")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚙️ Settings", use_container_width=True):
                st.switch_page("pages/0_setup.py")
        with col2:
            if st.button("⚙️ Config", use_container_width=True):
                st.switch_page("pages/settings.py")
        if st.button("📖 Help & Tutorial", use_container_width=True):
            st.switch_page("pages/6_help.py")

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
            # Keep the URL's authoritative ?mode= in sync with the toggle,
            # otherwise a sticky ?mode=owner (left by "Exit Helper Mode") would
            # be re-applied by init_session_state on the next run and fight the
            # toggle into a rerun loop.
            st.query_params["mode"] = "helper" if helper_mode else "owner"
            st.rerun()

    # Main content
    st.markdown('<p class="main-header">📷 PhotoPipe</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Photo Scanning Metadata Pipeline for Epson FastFoto</p>',
        unsafe_allow_html=True,
    )

    # Quick-start tutorial: expanded on a fresh install (nothing scanned yet),
    # collapsed but always available afterwards.
    st.markdown("---")
    fresh_install = not st.session_state.db.get_all_batches()
    with st.expander("🚀 Quick start — how PhotoPipe works", expanded=fresh_install):
        st.markdown(
            """
1. **📥 Capture** — feed stacks through the scanner into labeled *buckets*.
   Anyone can do this; flip on **Helper Mode** (sidebar) before handing over.
2. **🗂 Buckets** — convert each bucket to a *batch*, adding what you know:
   rough dates, place, people, event.
3. **👁️ Curate** — run AI dating; handwriting from photo backs is already
   read during capture. Review and approve.
4. **🧑 Faces** *(optional)* — group faces and name each person once; names
   become searchable keywords on every photo they appear in.
5. **✅ Finalize** — write the metadata into the files and export them
   organized by year.

Originals are always preserved untouched in the archive folder. The
**📖 Help** page walks through each step in detail.
            """
        )
        if st.button("Open the full tutorial", key="goto_help"):
            st.switch_page("pages/6_help.py")

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
