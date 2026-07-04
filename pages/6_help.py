"""Help & tutorial page: the full workflow, page by page, plus troubleshooting."""
import streamlit as st

st.set_page_config(page_title="Help - PhotoPipe", page_icon="📖", layout="wide")

st.title("📖 Help & Tutorial")
st.caption("How to get a shoebox of prints into a dated, tagged, searchable archive.")

st.markdown(
    """
## The big picture

PhotoPipe splits the work into two roles so anyone can help with the scanning
while you keep control of the metadata:

| Phase | Who | Where | What happens |
|---|---|---|---|
| **1 · Capture** | Anyone (a "helper") | 📥 Capture | Feed stacks through the scanner into labeled **buckets** |
| **2 · Curate** | You (the owner) | 🗂 Buckets → 📁 Batch Setup → 👁️ Curate → 🧑 Faces | Turn buckets into **batches**, add context, let AI estimate dates, name people |
| **3 · Finalize** | You | ✅ Finalize | Write EXIF/IPTC metadata and export organized, renamed files |

Your original scans are never touched: a pristine copy of every scan is
saved to the archive's `_originals` folder before any cropping happens.
"""
)

st.markdown("---")

with st.expander("**Step 1 — First-run Setup**", expanded=False):
    st.markdown(
        """
- Run once from the **Setup** page (the app sends you there automatically on first launch).
- Enter your **Anthropic API key** if you want AI dating and Claude-vision OCR
  fallback (optional — everything else works without it). A **Mistral API key**
  (set as the `MISTRAL_API_KEY` environment variable) enables cheap handwriting
  OCR on photo backs.
- Set your default location, family members, and copyright holder — these
  pre-fill new batches so you don't retype them.
- Run `photopipe doctor` in a terminal any time something seems off: it checks
  the scanner, ExifTool, and your API keys, and tells you how to fix what's broken.
"""
    )

with st.expander("**Step 2 — Capture: scan stacks into buckets**", expanded=False):
    st.markdown(
        """
1. On the **📥 Capture** page, type a bucket label that says where the stack
   came from — *"Grandma's blue album, page 3"* beats *"stack 1"*.
2. **Photograph the album cover or envelope** with the Mac's camera (the
   "Photo of the album or envelope" section — albums don't fit through the
   sheet feeder, so hold them up to the camera instead). Any handwriting or
   sticky notes on the cover get read by the AI later, so this photo often
   supplies the dates and event names by itself.
3. Load a stack of photos into the FastFoto's feeder (backs with handwriting
   are read automatically when duplex scanning is on) and hit **Scan Stack**.
   Repeat for the next stack — either into the same bucket or a new one.

**Owner tip:** before handing albums over, stick a Post-it on each cover
with whatever you know — *"≈1987–89, lake trips"*. The helper photographs
the cover, the AI reads your note, and the context arrives with the scans.

**Handing the scanner to a family member?** Toggle **Helper Mode** in the
home-page sidebar first: the app collapses to just the scan screen, so they
can't wander into your batches. To exit Helper Mode, go back to the home page URL.

Tips:
- A bucket = one physical origin (an album, an envelope, a shoebox layer).
  Keeping origins separate makes dating much easier later.
- Paper jam mid-stack? The photos scanned before the jam are kept and
  recorded — just reload the rest and scan again into the same bucket.
"""
    )

with st.expander("**Step 3 — Buckets → Batches: add what you know**", expanded=False):
    st.markdown(
        """
1. On the **🗂 Buckets** page, review what the helpers scanned.
2. Click **✨ Suggest context** on a bucket: one AI pass reads the album-cover
   photo (including your Post-its), rolls up the handwriting dates already
   found on photo backs, and looks at a sample of photos from across the
   whole bucket. It proposes a batch name, date range, locations, and a
   **list of distinct events** (one album often holds several). The proposal
   pre-fills the convert form — nothing is saved until you confirm.
3. Convert the bucket to a **batch**, correcting or adding anything the AI
   missed: date range, location, people, event. Rough is fine — *"summer
   1985, Lake Erie, the Kowalskis"* gives the AI a lot to work with.
4. Batches can also be created directly on **📁 Batch Setup** (and edited
   there later — dates are edited with proper date pickers, so nothing gets
   mangled).
"""
    )

with st.expander("**Step 4 — Curate: dates from handwriting + AI**", expanded=False):
    st.markdown(
        """
Dates are chosen in this priority order:

1. **Handwriting on the back** (highest trust) — read during capture by
   Mistral OCR, with a Claude-vision fallback for hard-to-read backs.
2. **Date stamps on the front** — orange lab-print stamps.
3. **AI estimation** — on the **👁️ Curate** page, click **Run AI dating**:
   Claude sees 10–15 photos per call so it can reason about them as a group
   ("same roll of film? same event? where does the timeline break?").
4. **Batch default** — photos with no other signal get spread across the
   batch's date range in scan order.

Review the results, correct anything that looks off, and approve. Photos you
haven't reviewed stay marked *needs review* and are skipped at finalize
(unless you choose otherwise).
"""
    )

with st.expander("**Step 5 — Faces (optional): name people once**", expanded=False):
    st.markdown(
        """
On the **🧑 Faces** page: **Detect faces** → **Group faces** → type each
person's name → **Apply names**. Every photo a person appears in gets their
name added to its keywords, which ends up in the exported files' metadata —
so your photo app can find *"Grandma Rose"* across the whole archive.

- The face model (~300 MB) downloads automatically on first use. All face
  data stays on this machine; nothing is sent to any API.
- One person split into two groups? Use **Merge groups**.
- A face in "Unsorted"? Use **Move an unsorted face into a group**.
- Renamed someone after applying? Just **Apply names** again — the old
  name is cleaned up automatically.
"""
    )

with st.expander("**Step 6 — Finalize: write metadata and export**", expanded=False):
    st.markdown(
        """
The **✅ Finalize** page writes each approved photo's date, GPS location,
people keywords, caption, and copyright into the file itself (EXIF/IPTC/XMP
via ExifTool), renames it (`PP_1985-06-01_Summer_1985_0001.jpg`), and files it
into `~/Pictures/Scanned_Photos/<year>/…`.

- Untouched originals are kept in the archive folder (on by default).
- A `_batch_report.json` lands next to the exported photos summarizing
  what was written and where each date came from.
- From there, import into Apple Photos or any other library — the metadata
  travels inside the files.
"""
    )

st.markdown("---")

st.markdown(
    """
## Troubleshooting

| Symptom | Fix |
|---|---|
| Scanner not detected | Power-cycle the FastFoto, then check **System Settings → Privacy & Security → Local Network** — macOS sometimes silently drops the permission after updates. Run `scanimage -L` in a terminal to re-prompt. |
| "AI Dating: no API key" | Enter the key on the **Setup** page (stored locally, `chmod 600`) or `export ANTHROPIC_API_KEY=…` before launching. |
| Handwriting OCR skipped | Set `MISTRAL_API_KEY`, or switch `handwriting_ocr.provider` to `claude` in Settings to use Claude vision only. |
| No batches on Curate | Convert a bucket to a batch on the **Buckets** page first. |
| Face model errors after an interrupted download | Delete `~/.insightface/models/buffalo_l` and run Detect again. |
| Stuck in Helper Mode | Open the home page URL directly; the toggle lives in the home sidebar. |
| Anything else | Run `photopipe doctor` in a terminal — it diagnoses the environment and prints the fix. |
"""
)

st.caption(
    "Config lives in `~/.photopipe/config.yaml`; the database in "
    "`~/.photopipe/photopipe.db`. Copy both (plus your Pictures folders) to move machines."
)
