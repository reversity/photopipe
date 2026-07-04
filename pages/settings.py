"""
Settings Page - Configure PhotoPipe settings.
"""

import os
import streamlit as st
from pathlib import Path

from photopipe.config import get_config, save_config, reload_config, Config
from photopipe.database import Database
from photopipe.metadata import check_exiftool_installed
from photopipe.vlm_client import is_vlm_available


st.set_page_config(page_title="Settings - PhotoPipe", page_icon="⚙️", layout="wide")


def init_session_state():
    """Initialize session state."""
    if "db" not in st.session_state:
        config = get_config()
        config.ensure_directories()
        st.session_state.db = Database()


def system_status():
    """Display system status and dependencies."""
    st.subheader("🔧 System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**ExifTool**")
        if check_exiftool_installed():
            st.success("✅ Installed")
        else:
            st.error("❌ Not installed")
            st.code("brew install exiftool")

    with col2:
        st.write("**Claude VLM**")
        if is_vlm_available():
            st.success("✅ Available")
        else:
            config = get_config()
            api_key = os.environ.get(config.vlm.api_key_env_var)
            if not api_key:
                st.warning("⚠️ No API key")
            else:
                st.warning("⚠️ anthropic package missing")


def path_settings():
    """Configure file paths."""
    st.subheader("📁 File Paths")

    config = get_config()

    with st.form("path_settings"):
        input_folder = st.text_input(
            "Scanner Input Folder",
            value=str(config.paths.input_folder),
            help="Where your scanner saves images",
        )

        output_folder = st.text_input(
            "Output Folder",
            value=str(config.paths.output_folder),
            help="Where finalized photos will be saved",
        )

        archive_folder = st.text_input(
            "Archive Folder",
            value=str(config.paths.archive_folder),
            help="Where original files are backed up",
        )

        database_path = st.text_input(
            "Database Path",
            value=str(config.paths.database),
            help="SQLite database location",
        )

        if st.form_submit_button("Save Path Settings"):
            old_database_path = config.paths.database

            config.paths.input_folder = Path(input_folder).expanduser()
            config.paths.output_folder = Path(output_folder).expanduser()
            config.paths.archive_folder = Path(archive_folder).expanduser()
            config.paths.database = Path(database_path).expanduser()

            save_config(config)
            config.ensure_directories()

            if config.paths.database != old_database_path:
                st.session_state.db = Database()
                st.info("Database path changed — reconnected to the new database.")

            st.success("✅ Path settings saved!")


def scanner_settings():
    """Configure scanner patterns."""
    st.subheader("📷 Scanner Settings")

    config = get_config()

    with st.form("scanner_settings"):
        st.write("**File Naming Patterns**")
        st.caption("Use {num} as placeholder for sequence number")

        front_pattern = st.text_input(
            "Front Image Pattern",
            value=config.scanner.front_pattern,
            help="Pattern for front images, e.g., IMG_{num}.jpg",
        )

        back_pattern = st.text_input(
            "Back Image Pattern",
            value=config.scanner.back_pattern,
            help="Pattern for back images, e.g., IMG_{num}_back.jpg",
        )

        watch_interval = st.number_input(
            "Watch Interval (seconds)",
            value=config.scanner.watch_interval_seconds,
            min_value=1,
            max_value=30,
            help="Time to wait for file to stabilize before processing",
        )

        if st.form_submit_button("Save Scanner Settings"):
            config.scanner.front_pattern = front_pattern
            config.scanner.back_pattern = back_pattern
            config.scanner.watch_interval_seconds = watch_interval

            save_config(config)
            st.success("✅ Scanner settings saved!")


def handwriting_ocr_settings():
    """Configure handwriting OCR (photo backs) settings."""
    st.subheader("🔤 Handwriting OCR Settings")

    config = get_config()

    with st.form("handwriting_ocr_settings"):
        provider_options = ["auto", "mistral", "claude"]
        current_provider = config.handwriting_ocr.provider
        try:
            provider_index = provider_options.index(current_provider)
        except ValueError:
            provider_index = 0

        provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=provider_index,
            help="'auto' uses Mistral first and falls back to Claude on low confidence.",
        )

        mistral_model = st.text_input(
            "Mistral Model",
            value=config.handwriting_ocr.mistral_model,
            help="Mistral OCR model identifier (e.g., mistral-ocr-3).",
        )

        mistral_max_image_dim = st.slider(
            "Mistral Max Image Dimension",
            min_value=1024,
            max_value=4096,
            value=config.handwriting_ocr.mistral_max_image_dim,
            step=256,
            help="Images resized to this dimension before submission to Mistral.",
        )

        confidence_fallback_threshold = st.slider(
            "Confidence Fallback Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(config.handwriting_ocr.confidence_fallback_threshold),
            step=0.05,
            help="Below this Mistral confidence, fall back to the Claude VLM.",
        )

        use_batch_api = st.checkbox(
            "Use Batch API (not yet implemented)",
            value=config.handwriting_ocr.use_batch_api,
            help="Reserved for a future release — OCR currently always runs "
                 "synchronously per photo, regardless of this setting.",
        )

        if st.form_submit_button("Save Handwriting OCR Settings"):
            config.handwriting_ocr.provider = provider
            config.handwriting_ocr.mistral_model = mistral_model
            config.handwriting_ocr.mistral_max_image_dim = mistral_max_image_dim
            config.handwriting_ocr.confidence_fallback_threshold = confidence_fallback_threshold
            config.handwriting_ocr.use_batch_api = use_batch_api

            save_config(config)
            st.success("✅ Handwriting OCR settings saved!")


def vlm_settings():
    """Configure Claude vision-language model settings."""
    st.subheader("🤖 Claude VLM Settings")

    config = get_config()

    # API key status: env var first, then the key stored by the setup wizard
    env_key = os.environ.get(config.vlm.api_key_env_var)
    api_key = config.get_api_key()
    if env_key:
        st.success(f"✅ API key found in environment ({config.vlm.api_key_env_var})")
        st.caption(f"Key: {env_key[:8]}...{env_key[-4:]}")
    elif api_key:
        st.success("✅ API key found in PhotoPipe settings (saved by the Setup wizard)")
        st.caption(f"Key: {api_key[:8]}...{api_key[-4:]}")
    else:
        st.warning(f"⚠️ No API key found in {config.vlm.api_key_env_var} or PhotoPipe settings")
        st.info("""
        To enable the Claude VLM, enter your key on the **Setup** page, or set it in your shell:
        ```bash
        export ANTHROPIC_API_KEY='your-key-here'
        ```
        Then restart PhotoPipe.
        """)

    with st.form("vlm_settings"):
        model = st.text_input(
            "Model",
            value=config.vlm.model,
            help="Claude model alias used for vision calls.",
        )
        st.caption("Run `python -m photopipe doctor` to verify the model alias is valid.")

        max_image_dimension = st.slider(
            "Max Image Dimension",
            min_value=512,
            max_value=2048,
            value=config.vlm.max_image_dimension,
            step=256,
            help="Images resized to this dimension to manage vision token cost.",
        )

        cache_ttl_options = ["5m", "1h"]
        current_ttl = config.vlm.cache_ttl
        try:
            cache_ttl_index = cache_ttl_options.index(current_ttl)
        except ValueError:
            cache_ttl_index = 0
        cache_ttl = st.selectbox(
            "Prompt Cache TTL",
            options=cache_ttl_options,
            index=cache_ttl_index,
            help="How long the cached prompt prefix should live.",
        )

        batch_api_threshold = st.number_input(
            "Batch API Threshold (not yet implemented)",
            min_value=1,
            value=config.vlm.batch_api_threshold,
            help="Reserved for a future release — AI dating currently always "
                 "runs synchronously, regardless of this setting.",
        )

        if st.form_submit_button("Save Claude VLM Settings"):
            config.vlm.model = model
            config.vlm.max_image_dimension = max_image_dimension
            config.vlm.cache_ttl = cache_ttl
            config.vlm.batch_api_threshold = int(batch_api_threshold)

            save_config(config)
            st.success("✅ Claude VLM settings saved!")


def output_settings():
    """Configure output settings."""
    st.subheader("📤 Output Settings")

    config = get_config()

    with st.form("output_settings"):
        folder_structure = st.text_input(
            "Folder Structure Template",
            value=config.output.folder_structure,
            help="Use {year}, {month}, {batch_name}",
        )

        filename_template = st.text_input(
            "Filename Template",
            value=config.output.filename_template,
            help="Use {date}, {batch_name}, {sequence}, {side}",
        )

        col1, col2 = st.columns(2)

        with col1:
            preserve = st.checkbox(
                "Preserve Original Files",
                value=config.output.preserve_originals,
                help="Copy originals to archive folder",
            )

        with col2:
            web_copies = st.checkbox(
                "Generate Web Copies",
                value=config.output.generate_web_copies,
                help="Create resized versions for web",
            )

        web_max = st.slider(
            "Web Copy Max Dimension",
            min_value=1024,
            max_value=4096,
            value=config.output.web_copy_max_dimension,
            step=256,
            disabled=not config.output.generate_web_copies,
        )

        if st.form_submit_button("Save Output Settings"):
            config.output.folder_structure = folder_structure
            config.output.filename_template = filename_template
            config.output.preserve_originals = preserve
            config.output.generate_web_copies = web_copies
            config.output.web_copy_max_dimension = web_max

            save_config(config)
            st.success("✅ Output settings saved!")


def metadata_settings():
    """Configure default metadata settings."""
    st.subheader("📝 Metadata Defaults")

    config = get_config()

    with st.form("metadata_settings"):
        timezone = st.text_input(
            "Default Timezone",
            value=config.metadata.default_timezone,
            help="Timezone for dates without time (e.g., America/New_York)",
        )

        copyright_template = st.text_input(
            "Copyright Template",
            value=config.metadata.copyright_template,
            help="Use {year} for the photo year",
        )

        if st.form_submit_button("Save Metadata Settings"):
            config.metadata.default_timezone = timezone
            config.metadata.copyright_template = copyright_template

            save_config(config)
            st.success("✅ Metadata settings saved!")


def database_management():
    """Database management tools."""
    st.subheader("🗄️ Database Management")

    db = st.session_state.db
    config = get_config()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Database Location:**")
        st.code(str(config.paths.database))

        # Stats
        batches = db.get_all_batches()
        st.write(f"- Batches: {len(batches)}")

        total_photos = sum(db.get_batch_photo_count(b.id) for b in batches)
        st.write(f"- Total Photos: {total_photos}")

    with col2:
        st.write("**Actions:**")

        if st.button("🔄 Reload Configuration"):
            reload_config()
            st.success("Configuration reloaded!")
            st.rerun()

        st.write("")

        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ This will delete ALL batches and photos!")
            if st.button("Yes, I'm sure", key="confirm_clear_yes"):
                for batch in batches:
                    db.delete_batch(batch.id)
                st.session_state.confirm_clear = False
                st.success("All data cleared!")
                st.rerun()
            if st.button("Cancel", key="confirm_clear_cancel"):
                st.session_state.confirm_clear = False
                st.rerun()
        else:
            if st.button("🗑️ Clear All Data", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()


def export_import_config():
    """Export/import configuration."""
    st.subheader("📦 Export/Import Configuration")

    config = get_config()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Export Configuration**")
        import yaml
        config_yaml = yaml.dump(config.model_dump(mode="json"), default_flow_style=False)
        st.download_button(
            label="📥 Download config.yaml",
            data=config_yaml,
            file_name="photopipe_config.yaml",
            mime="text/yaml",
        )

    with col2:
        st.write("**Import Configuration**")
        uploaded = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
        if uploaded:
            import yaml
            try:
                data = yaml.safe_load(uploaded.read())
                new_config = Config(**data)
                save_config(new_config)
                st.success("Configuration imported!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import: {e}")


def main():
    """Main page."""
    init_session_state()

    st.title("⚙️ Settings")
    st.write("Configure PhotoPipe settings and preferences.")

    # System status at top
    system_status()

    st.markdown("---")

    # Settings in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Paths",
        "📷 Scanner",
        "🔤 Handwriting OCR",
        "🤖 Claude VLM",
        "📤 Output",
        "🗄️ Database",
    ])

    with tab1:
        path_settings()

    with tab2:
        scanner_settings()

    with tab3:
        handwriting_ocr_settings()

    with tab4:
        vlm_settings()

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            output_settings()
        with col2:
            metadata_settings()

    with tab6:
        database_management()
        st.markdown("---")
        export_import_config()

    # Navigation
    st.markdown("---")
    if st.button("← Back to Home"):
        st.switch_page("app.py")


if __name__ == "__main__":
    main()
