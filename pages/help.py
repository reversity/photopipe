"""
Help & Tutorial Page - Guide users through PhotoPipe.
"""

import streamlit as st

st.set_page_config(
    page_title="Help - PhotoPipe",
    page_icon="❓",
    layout="wide",
)


def workflow_overview():
    """Show the overall workflow."""
    st.header("📋 Workflow Overview")

    st.markdown("""
    PhotoPipe helps you scan, organize, and add metadata to old photos.
    Here's the typical workflow:
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ### 1️⃣ Scan
        **Batch Setup Page**

        - Connect your Epson FastFoto
        - Scan photos (front & back)
        - Photos save to input folder
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ Organize
        **Batch Setup Page**

        - Create a batch with date/location
        - Group related photos together
        - Add people who appear in photos
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ Process
        **Scan Ingest Page**

        - Import scans into batch
        - OCR reads dates from photo backs
        - AI estimates dates from image content
        """)

    with col4:
        st.markdown("""
        ### 4️⃣ Export
        **Finalize Page**

        - Review & correct metadata
        - Embed into photo files
        - Organize into dated folders
        """)


def quick_start():
    """Quick start guide."""
    st.header("🚀 Quick Start Guide")

    with st.expander("**Step 1: Scan Your Photos**", expanded=True):
        st.markdown("""
        1. Go to **Batch Setup** in the sidebar
        2. Connect your Epson FastFoto scanner (USB or WiFi)
        3. Load photos into the scanner's document feeder
        4. Click **Start Scanning**

        💡 **Tip:** Enable "Scan Backs (Duplex)" to capture handwritten dates and notes on photo backs.

        📁 Scanned photos are saved to: `~/Pictures/Scanner_Input`
        """)

    with st.expander("**Step 2: Create a Batch**"):
        st.markdown("""
        A "batch" is a group of related photos - like photos from the same event, trip, or time period.

        1. Enter a descriptive **Batch Name** (e.g., "Summer_Vacation_1985")
        2. Enter an **Approximate Date** - this can be:
           - Just a year: `1985`
           - Month and year: `June 1985`
           - Season: `Summer 1985`
           - Approximate: `Early 1990s`
        3. Add **Location** (city, state or address)
        4. List **People** in the photos (comma-separated)
        5. Click **Create Batch**

        💡 **Tip:** You don't need exact dates! PhotoPipe will try to extract dates from photo backs using OCR, or estimate them using AI.
        """)

    with st.expander("**Step 3: Ingest & Process**"):
        st.markdown("""
        1. Go to **Scan Ingest** in the sidebar
        2. Select your batch from the dropdown
        3. Click **Scan & Ingest Photos** to import scans
        4. Click **Run OCR** to extract dates from photo backs
        5. (Optional) Click **Estimate Dates with AI** for photos without dates

        💡 **Tip:** The AI analyzes clothing, hairstyles, cars, and other visual clues to estimate when photos were taken.
        """)

    with st.expander("**Step 4: Review**"):
        st.markdown("""
        1. Go to **Review** in the sidebar
        2. Review each photo's metadata
        3. Correct any dates or add descriptions
        4. Mark photos as reviewed

        💡 **Tip:** Photos flagged with ⚠️ have low-confidence OCR and need your attention.
        """)

    with st.expander("**Step 5: Finalize & Export**"):
        st.markdown("""
        1. Go to **Finalize** in the sidebar
        2. Review the export preview
        3. Click **Finalize Batch**

        PhotoPipe will:
        - Embed metadata (date, location, people) into each photo
        - Organize photos into folders by year/month
        - Archive the original scans

        📁 Finished photos are saved to: `~/Pictures/Scanned_Photos`
        """)


def tips_and_tricks():
    """Tips and tricks section."""
    st.header("💡 Tips & Tricks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Getting Better OCR Results

        - **Clean your scanner glass** - dust causes errors
        - **Flatten curled photos** - creases confuse OCR
        - **Use good lighting** - faded text is hard to read
        - **Write dates clearly** - block letters work best

        ### AI Date Estimation

        The AI looks for visual clues like:
        - 👔 Clothing and fashion styles
        - 💇 Hairstyles
        - 🚗 Vehicles (cars are great date markers!)
        - 📺 Technology (TVs, phones, computers)
        - 🏠 Interior design and decor
        """)

    with col2:
        st.markdown("""
        ### Organizing Tips

        - **Batch by event** - "Christmas 1987", "Beach Trip 1992"
        - **Use consistent names** - makes searching easier
        - **Add everyone's names** - even if not in every photo
        - **Be approximate** - "Summer 1985" is fine if you're unsure

        ### Scanner Tips (Epson FastFoto)

        - **600 DPI** is best for most photos
        - **Enable duplex** to capture backs automatically
        - **Use the ADF** for speed, flatbed for fragile photos
        - **Clean the rollers** monthly for smooth feeding
        """)


def faq():
    """Frequently asked questions."""
    st.header("❓ FAQ")

    with st.expander("What if I don't know the exact date?"):
        st.markdown("""
        That's totally fine! Enter whatever you know:
        - Just the year: `1985`
        - A season: `Summer 1985`
        - A range: `Early 1980s`

        PhotoPipe will try to narrow it down using OCR and AI. If all else fails, photos will be dated within your specified range.
        """)

    with st.expander("What metadata gets embedded?"):
        st.markdown("""
        PhotoPipe embeds standard EXIF/IPTC/XMP metadata that works with all photo software:
        - **Date Taken** - When the photo was taken
        - **GPS Coordinates** - Where the photo was taken
        - **Description** - Event description
        - **Keywords** - People's names, location
        - **Copyright** - Your copyright notice
        """)

    with st.expander("What happens to my original scans?"):
        st.markdown("""
        Your originals are safe! PhotoPipe:
        1. Copies originals to an `_archive` folder
        2. Works on copies, not originals
        3. Never deletes your scans

        Archive location: `~/Pictures/Scanned_Photos/_archive`
        """)

    with st.expander("Can I process photos that weren't scanned with FastFoto?"):
        st.markdown("""
        Yes! Just copy your photos to the input folder:
        `~/Pictures/Scanner_Input`

        Then create a batch and ingest them normally. They don't need to follow the FastFoto naming convention.
        """)

    with st.expander("How does the AI dating work?"):
        st.markdown("""
        PhotoPipe uses Claude (Anthropic's AI) to analyze visual elements in your photos:

        1. It selects a few representative photos from your batch
        2. Analyzes clothing, hairstyles, technology, cars, etc.
        3. Estimates a likely date range
        4. You can apply this estimate to all undated photos

        **Note:** You need an Anthropic API key for this feature (set in Settings).
        """)

    with st.expander("What if the scanner isn't detected?"):
        st.markdown("""
        If your scanner isn't showing up:

        1. **Check connection** - USB or WiFi connected?
        2. **Power cycle** - Turn scanner off and on
        3. **Install drivers** - Get latest from Epson support
        4. **Use Epson software** - Scan with FastFoto software, then import

        PhotoPipe will work with any scanned photos - scanner control is optional.
        """)


def keyboard_shortcuts():
    """Keyboard shortcuts reference."""
    st.header("⌨️ Keyboard Shortcuts")

    st.markdown("""
    | Shortcut | Action |
    |----------|--------|
    | `R` | Refresh page |
    | `←` / `→` | Navigate photos in review |
    | `Enter` | Confirm/Submit |
    | `Esc` | Cancel/Close |
    """)


def main():
    st.title("❓ Help & Tutorial")

    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "📋 Workflow", "💡 Tips", "❓ FAQ"])

    with tab1:
        quick_start()

    with tab2:
        workflow_overview()

    with tab3:
        tips_and_tricks()

    with tab4:
        faq()

    st.markdown("---")
    keyboard_shortcuts()

    st.markdown("---")
    st.caption("Need more help? Check the README or file an issue on GitHub.")


if __name__ == "__main__":
    main()
