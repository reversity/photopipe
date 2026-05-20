"""Owner curate page: pick converted batch, run AI, review results."""
import streamlit as st
from photopipe.config import get_config
from photopipe.curate_pipeline import run_ai_dating, apply_ai_results
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import PhotoPhase

st.set_page_config(page_title="Curate - PhotoPipe", page_icon="🧬", layout="wide")


def init_state():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    st.session_state.setdefault("curate_batch_id", None)
    st.session_state.setdefault("ai_run_result", None)


def main():
    init_state()
    db = st.session_state.db
    st.title("🧬 Curate")
    st.caption("Apply AI dating and review estimates for converted batches.")

    # Batch selector — show only batches with photos still awaiting finalization.
    # Note: PhotoPair has use_enum_values=True, so photo.phase is a plain string
    # (e.g. "captured", "curated", "finalized"). Comparisons against
    # PhotoPhase.X still work because PhotoPhase is a str-Enum mixin.
    batches = [b for b in db.get_all_batches()
               if any(p.phase != PhotoPhase.FINALIZED
                      for p in db.get_photos_by_batch(b.id))]
    if not batches:
        st.info("No batches awaiting curation. Convert a bucket from the Buckets page.")
        return

    names = [b.name for b in batches]
    chosen = st.selectbox("Batch", names)
    batch = next(b for b in batches if b.name == chosen)
    photos = db.get_photos_by_batch(batch.id)

    tab1, tab2, tab3 = st.tabs(["🤖 AI Dating", "🔍 Review", "ℹ️ Context"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Photos", len(photos))
            st.metric("Already dated", sum(1 for p in photos if p.extracted_date))
        with col2:
            images_per_call = st.slider("Photos per AI call", 5, 20, 12)
            st.caption(
                "Runs in realtime (synchronous). Batch API mode "
                "(async, 50% cheaper) is a planned follow-up."
            )

        if st.button("🤖 Run AI Dating", type="primary"):
            undated = [p for p in photos if not p.extracted_date]
            with st.spinner(f"Analyzing {len(undated)} photos..."):
                result = run_ai_dating(batch, undated, images_per_call=images_per_call)
                st.session_state.ai_run_result = result
            st.success(f"Analyzed {len(undated)} photos in {len(result.raw_responses)} call(s)")
            st.rerun()

        # Show AI results
        ai = st.session_state.ai_run_result
        if ai:
            st.markdown("### Results")
            st.write("**Coherence:**", ai.coherence.get("summary", ""))
            if ai.coherence.get("segment_breaks"):
                st.warning(f"AI detected {len(ai.coherence['segment_breaks'])} segment break(s)")
                for sb in ai.coherence["segment_breaks"]:
                    st.write(f"- After photo {sb['after_photo_index']}: {sb['reason']}")
            if st.button("✅ Apply AI dates"):
                applied = apply_ai_results(batch, ai, photos, db=db)
                st.success(f"Updated {applied.updated} photos (skipped {applied.skipped})")
                st.session_state.ai_run_result = None
                st.rerun()

    with tab2:
        cols = st.columns(5)
        for i, photo in enumerate(photos):
            with cols[i % 5]:
                try:
                    st.image(generate_thumbnail(photo.front_path, size=(150, 150)))
                except Exception:
                    pass
                date_str = f"📅{photo.extracted_date.year}" if photo.extracted_date else "—"
                st.caption(f"#{photo.sequence_num} {date_str}")

    with tab3:
        st.write(f"**Name:** {batch.name}")
        st.write(f"**Date range:** {batch.date_start} – {batch.date_end}")
        st.write(f"**Location:** {batch.location_description or '—'}")
        st.write(f"**People:** {', '.join(batch.people) if batch.people else '—'}")
        st.write(f"**Event:** {batch.event_description or '—'}")


if __name__ == "__main__":
    main()
