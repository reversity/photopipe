"""
Settings Page - Configure PhotoPipe settings.
"""

import os
import streamlit as st
from pathlib import Path

from photopipe.config import get_config, save_config, reload_config, Config
from photopipe.database import Database
from photopipe.metadata import check_exiftool_installed
from photopipe.ai_dating import is_ai_dating_available


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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**ExifTool**")
        if check_exiftool_installed():
            st.success("✅ Installed")
        else:
            st.error("❌ Not installed")
            st.code("brew install exiftool")

    with col2:
        st.write("**Tesseract OCR**")
        import shutil
        if shutil.which("tesseract"):
            st.success("✅ Installed")
        else:
            st.error("❌ Not installed")
            st.code("brew install tesseract")

    with col3:
        st.write("**AI Dating**")
        if is_ai_dating_available():
            st.success("✅ Available")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
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
            config.paths.input_folder = Path(input_folder).expanduser()
            config.paths.output_folder = Path(output_folder).expanduser()
            config.paths.archive_folder = Path(archive_folder).expanduser()
            config.paths.database = Path(database_path).expanduser()

            save_config(config)
            config.ensure_directories()
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


def ocr_settings():
    """Configure OCR settings."""
    st.subheader("🔤 OCR Settings")

    config = get_config()

    with st.form("ocr_settings"):
        language = st.text_input(
            "Tesseract Language",
            value=config.ocr.language,
            help="Tesseract language code (eng, deu, fra, etc.)",
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0,
            max_value=100,
            value=config.ocr.confidence_threshold,
            help="Flag for review below this confidence level",
        )

        st.write("**Preprocessing**")
        col1, col2, col3 = st.columns(3)

        with col1:
            grayscale = st.checkbox(
                "Grayscale",
                value=config.ocr.preprocessing.grayscale,
            )

        with col2:
            threshold = st.checkbox(
                "Adaptive Threshold",
                value=config.ocr.preprocessing.adaptive_threshold,
            )

        with col3:
            deskew = st.checkbox(
                "Deskew",
                value=config.ocr.preprocessing.deskew,
            )

        if st.form_submit_button("Save OCR Settings"):
            config.ocr.language = language
            config.ocr.confidence_threshold = confidence_threshold
            config.ocr.preprocessing.grayscale = grayscale
            config.ocr.preprocessing.adaptive_threshold = threshold
            config.ocr.preprocessing.deskew = deskew

            save_config(config)
            st.success("✅ OCR settings saved!")


def ai_settings():
    """Configure AI dating settings."""
    st.subheader("🤖 AI Dating Settings")

    config = get_config()

    # API key status
    api_key = os.environ.get(config.ai_dating.api_key_env_var)
    if api_key:
        st.success(f"✅ API key found in environment ({config.ai_dating.api_key_env_var})")
        st.caption(f"Key: {api_key[:8]}...{api_key[-4:]}")
    else:
        st.warning(f"⚠️ No API key found in {config.ai_dating.api_key_env_var}")
        st.info("""
        To enable AI dating, set your Anthropic API key:
        ```bash
        export ANTHROPIC_API_KEY='your-key-here'
        ```
        Then restart PhotoPipe.
        """)

    with st.form("ai_settings"):
        enabled = st.checkbox(
            "Enable AI Dating",
            value=config.ai_dating.enabled,
        )

        model = st.selectbox(
            "Model",
            options=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"],
            index=0 if config.ai_dating.model == "claude-sonnet-4-20250514" else 1,
        )

        max_samples = st.slider(
            "Max Samples per Batch",
            min_value=1,
            max_value=10,
            value=config.ai_dating.max_samples_per_batch,
            help="Number of representative photos to analyze",
        )

        max_dimension = st.slider(
            "Max Image Dimension",
            min_value=512,
            max_value=2048,
            value=config.ai_dating.max_image_dimension,
            step=256,
            help="Images resized to this dimension to save API costs",
        )

        if st.form_submit_button("Save AI Settings"):
            config.ai_dating.enabled = enabled
            config.ai_dating.model = model
            config.ai_dating.max_samples_per_batch = max_samples
            config.ai_dating.max_image_dimension = max_dimension

            save_config(config)
            st.success("✅ AI settings saved!")


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

        if st.button("🗑️ Clear All Data", type="secondary"):
            st.warning("⚠️ This will delete ALL batches and photos!")
            if st.button("Yes, I'm sure", key="confirm_clear"):
                for batch in batches:
                    db.delete_batch(batch.id)
                st.success("All data cleared!")
                st.rerun()


def export_import_config():
    """Export/import configuration."""
    st.subheader("📦 Export/Import Configuration")

    config = get_config()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Export Configuration**")
        if st.button("Download config.yaml"):
            import yaml
            config_yaml = yaml.dump(config.model_dump(), default_flow_style=False)
            st.download_button(
                label="📥 Download",
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
        "🔤 OCR",
        "🤖 AI",
        "📤 Output",
        "🗄️ Database",
    ])

    with tab1:
        path_settings()

    with tab2:
        scanner_settings()

    with tab3:
        ocr_settings()

    with tab4:
        ai_settings()

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
