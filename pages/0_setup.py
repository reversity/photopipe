"""
Setup Wizard - First-run configuration for PhotoPipe.
"""

import streamlit as st

from photopipe.setup import (
    load_settings,
    save_settings,
    is_setup_complete,
    UserSettings,
)


st.set_page_config(
    page_title="Setup - PhotoPipe",
    page_icon="⚙️",
    layout="centered",
)


def setup_wizard():
    """Render the setup wizard."""
    st.title("⚙️ PhotoPipe Setup")

    settings = load_settings()

    if settings.setup_complete:
        st.success("Setup is complete! You can update your settings below.")
    else:
        st.info("Welcome to PhotoPipe! Let's configure a few things to get started.")

    st.markdown("---")

    with st.form("setup_form"):
        st.subheader("API Configuration")

        api_key = st.text_input(
            "Anthropic API Key",
            value=settings.anthropic_api_key or "",
            type="password",
            help="Required for AI-powered date estimation. Get one at console.anthropic.com",
            placeholder="sk-ant-api03-...",
        )

        if api_key:
            st.caption("✓ API key entered")
        else:
            st.caption("Optional - AI dating will be disabled without this")

        st.markdown("---")
        st.subheader("Default Values")
        st.caption("These will be pre-filled when creating new batches")

        default_location = st.text_input(
            "Default Location",
            value=settings.default_location or "",
            placeholder="e.g., Toledo, OH",
            help="Your typical photo location (can be changed per batch)",
        )

        default_people = st.text_input(
            "Family Members / Common People",
            value=", ".join(settings.default_people) if settings.default_people else "",
            placeholder="e.g., Mom, Dad, Grandma Rose, Johnny, Sarah",
            help="Comma-separated list of people often in your photos",
        )

        copyright_holder = st.text_input(
            "Copyright Holder",
            value=settings.copyright_holder or "",
            placeholder="e.g., The Smith Family",
            help="Name to use in photo copyright metadata",
        )

        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            submitted = st.form_submit_button("Save Settings", type="primary")

        with col2:
            if settings.setup_complete:
                skip = st.form_submit_button("Cancel")
                if skip:
                    st.switch_page("pages/1_batch_setup.py")

        if submitted:
            # Parse people list
            people_list = [p.strip() for p in default_people.split(",") if p.strip()]

            # Check if this is first-time setup
            is_first_run = not settings.setup_complete

            # Update settings
            settings.anthropic_api_key = api_key if api_key else None
            settings.default_location = default_location if default_location else None
            settings.default_people = people_list
            settings.copyright_holder = copyright_holder if copyright_holder else None
            settings.setup_complete = True

            save_settings(settings)

            st.session_state.settings_saved = True
            st.session_state.is_first_run = is_first_run
            st.rerun()

    # Show post-save UI outside the form
    if st.session_state.get("settings_saved"):
        st.success("Settings saved!")

        if st.session_state.get("is_first_run"):
            st.info("🎉 Setup complete! Would you like to see the tutorial?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📖 View Tutorial", type="primary"):
                    st.session_state.settings_saved = False
                    st.switch_page("pages/help.py")
            with col2:
                if st.button("➡️ Start Using PhotoPipe"):
                    st.session_state.settings_saved = False
                    st.switch_page("pages/1_batch_setup.py")
        else:
            st.session_state.settings_saved = False
            st.switch_page("pages/1_batch_setup.py")


def main():
    setup_wizard()


if __name__ == "__main__":
    main()
