"""Owner-facing bucket dashboard: list, view stats, suggest context, convert to batch."""
from datetime import date

import streamlit as st
from photopipe.bucket_service import BucketService
from photopipe.config import get_config
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import BucketStatus
from photopipe.vlm_client import is_vlm_available

st.set_page_config(page_title="Buckets - PhotoPipe", page_icon="📦", layout="wide")

# Family photos can predate 1900; st.date_input raises if a pre-filled value
# falls outside [min, max], so bound generously and clamp pre-fills into range.
MIN_PHOTO_DATE = date(1826, 1, 1)  # oldest surviving photograph


def _today() -> date:
    return date.today()


def _clamp_date(d):
    if d is None:
        return None
    if d < MIN_PHOTO_DATE:
        return MIN_PHOTO_DATE
    if d > _today():
        return _today()
    return d


def _iso_to_date(value):
    try:
        return _clamp_date(date.fromisoformat(value)) if value else None
    except (ValueError, TypeError):
        return None


def _suggestion_defaults(bucket) -> dict:
    """Convert-form defaults drawn from the AI proposal (owner always edits)."""
    sc = bucket.suggested_context or {}
    events = sc.get("events") or []
    event_text = "\n".join(
        f"- {e.get('description', '')}"
        + (f" (~{e['approx_date']})" if e.get("approx_date") else "")
        for e in events
        if e.get("description")
    )
    locations = " / ".join(sc.get("location_guesses") or [])
    dr = sc.get("date_range") or {}
    return {
        "name": sc.get("suggested_batch_name") or bucket.label,
        "date_start": _iso_to_date(dr.get("start")),
        "date_end": _iso_to_date(dr.get("end")),
        "location": locations,
        "event": event_text,
    }


def _show_suggestion(sc: dict) -> None:
    """Render the AI proposal so the owner can judge it before converting."""
    conf = sc.get("confidence", "low")
    badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
    st.markdown(f"**✨ AI-suggested context** {badge} *{conf} confidence*")
    if sc.get("container_text"):
        st.markdown(f"**Read off the album/envelope:** {sc['container_text']}")
    dr = sc.get("date_range") or {}
    rollup = sc.get("ocr_date_rollup") or {}
    if dr.get("start") or dr.get("end"):
        source = (
            f" (from {rollup['count']} handwritten dates on photo backs)"
            if rollup.get("count")
            else " (visual estimate)"
        )
        st.markdown(f"**Dates:** {dr.get('start') or '?'} → {dr.get('end') or '?'}{source}")
    if sc.get("era_guess"):
        st.markdown(f"**Era:** {sc['era_guess']}")
    events = sc.get("events") or []
    if events:
        st.markdown("**Events found** (an album often holds several):")
        for e in events:
            approx = f" · ~{e['approx_date']}" if e.get("approx_date") else ""
            st.markdown(f"- {e.get('description', '?')}{approx} · *{e.get('confidence', '?')}*")
    if sc.get("location_guesses"):
        st.markdown(f"**Location clues:** {', '.join(sc['location_guesses'])}")
    if sc.get("reasoning"):
        st.caption(f"Why: {sc['reasoning']}")


def main():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📦 Buckets")
    st.caption("Raw scans waiting to be curated into batches.")

    show_all = st.checkbox("Show converted buckets", value=False, help="Buckets that have already become batches are hidden by default. Check this to see them for reference — they can't be converted again.")
    # Note: Bucket has use_enum_values=True, so bucket.status is a plain string
    # (e.g. "open", "closed", "converted"). Comparisons against BucketStatus.X
    # still work because BucketStatus is a str-Enum mixin.
    buckets = svc.db.list_buckets() if show_all else [
        b for b in svc.db.list_buckets() if b.status != BucketStatus.CONVERTED
    ]

    if not buckets:
        st.info("No open buckets. Switch to Helper Mode and scan some photos.")
        return

    for bucket in buckets:
        stats = svc.get_stats(bucket.id)
        with st.expander(
            f"📁 {bucket.label}  ·  {stats.photo_count} photos  "
            f"·  {'helper: ' + stats.helper_name if stats.helper_name else 'unattributed'}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.metric("Photos", stats.photo_count)
            with col2:
                st.metric("With dates from back", stats.photos_with_extracted_date)
            with col3:
                # bucket.status is already a string thanks to use_enum_values
                st.write(f"**Status:** {bucket.status}")

            # Re-run crop/deskew from the pristine originals (e.g. to apply the
            # improved deskew to already-scanned photos).
            from photopipe.capture_pipeline import reprocess_bucket, background_pending
            pending = background_pending(bucket.id)
            if pending:
                st.caption(f"⚙️ Re-processing {pending} photo(s) in the background…")
            elif st.button(
                "🔄 Re-crop from originals", key=f"reproc_{bucket.id}",
                help="Re-runs auto-crop/deskew on this bucket's photos from their "
                     "untouched originals, applying the current settings. Safe to "
                     "run repeatedly — it always works from the pristine copy.",
            ):
                n = reprocess_bucket(db, bucket.id)
                if n:
                    st.success(f"Re-processing {n} photos in the background…")
                    st.rerun()
                else:
                    st.warning("No pristine originals found for this bucket (older scans).")

            # Container photo (album cover / envelope) + thumbnail strip
            photos = db.get_photos_by_bucket(bucket.id)
            if bucket.context_image_path and bucket.context_image_path.exists():
                cc1, cc2 = st.columns([1, 4])
                with cc1:
                    st.image(str(bucket.context_image_path), width=160)
                    st.caption("Album / envelope")
                with cc2:
                    if photos:
                        cols = st.columns(6)
                        for i, p in enumerate(photos[:6]):
                            with cols[i]:
                                try:
                                    st.image(generate_thumbnail(p.front_path, size=(120, 120)))
                                except Exception:
                                    st.caption(f"#{p.sequence_num}")
            elif photos:
                cols = st.columns(6)
                for i, p in enumerate(photos[:6]):
                    with cols[i]:
                        try:
                            st.image(generate_thumbnail(p.front_path, size=(120, 120)))
                        except Exception:
                            st.caption(f"#{p.sequence_num}")

            # AI context triage: propose dates/events/locations before the
            # owner types anything.
            if bucket.status != BucketStatus.CONVERTED and photos:
                if st.button(
                    "✨ Suggest context",
                    key=f"suggest_{bucket.id}",
                    disabled=not is_vlm_available(),
                    help="One Claude vision call over the album cover photo and a "
                         "sample of photos from across the bucket, plus the "
                         "handwriting dates already read from the backs. Pre-fills "
                         "the form below — you confirm or edit everything."
                    if is_vlm_available()
                    else "Needs an Anthropic API key (Setup page).",
                ):
                    from photopipe.bucket_triage import suggest_bucket_context

                    with st.spinner("Reading the album cover and sampling photos…"):
                        try:
                            suggest_bucket_context(bucket, db)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Suggestion failed: {e}")

                if bucket.suggested_context:
                    _show_suggestion(bucket.suggested_context)

            if bucket.status == BucketStatus.CLOSED:
                defaults = _suggestion_defaults(bucket)
                suggested = bool(bucket.suggested_context)
                with st.form(f"convert_{bucket.id}"):
                    st.markdown("### Convert to Batch")
                    if suggested:
                        st.caption("Pre-filled from the AI suggestion — check and correct before converting.")
                    name = st.text_input("Batch name", value=defaults["name"], help="Becomes the batch's name everywhere else in the app, including exported filenames. You can rename it later on the Batch Setup page.")
                    c1, c2 = st.columns(2)
                    with c1:
                        date_start = st.date_input("Date start (optional)", value=defaults["date_start"], min_value=MIN_PHOTO_DATE, max_value=_today(), help="Earliest date these photos could be from. Gives AI dating a range to work within — leave blank if you have no idea.")
                    with c2:
                        date_end = st.date_input("Date end (optional)", value=defaults["date_end"], min_value=MIN_PHOTO_DATE, max_value=_today(), help="Latest date these photos could be from. Leave blank if unknown.")
                    location = st.text_input("Location (optional)", value=defaults["location"], help="Where the photos were taken, e.g. 'Toledo, OH'. Saved with the batch and later written into each photo's metadata.")
                    people = st.text_input("People (comma-separated, optional)", help="Names of people likely in these photos, e.g. 'Mom, Dad, Grandma Rose'. Tagged onto every photo in the batch.")
                    event = st.text_area("Event description (optional)", value=defaults["event"], height=100 if defaults["event"] else 60, help="Any context you remember — the occasion, kids' ages, who scanned it. An album often spans several events; list them all, the AI uses this to place segment breaks.")
                    if st.form_submit_button("✅ Convert to Batch", type="primary", help="Moves this bucket's photos into a new batch so you can run AI dating on the Curate page. A bucket can only be converted once."):
                        batch = svc.convert_to_batch(
                            bucket.id,
                            name=name,
                            date_start=date_start or None,
                            date_end=date_end or None,
                            location_description=location or None,
                            people=[p.strip() for p in people.split(",") if p.strip()],
                            event_description=event or None,
                        )
                        st.success(f"Converted to batch '{batch.name}'")
                        st.rerun()

            if bucket.status == BucketStatus.CONVERTED:
                st.info(f"Already converted to batch (id: {bucket.batch_id})")


if __name__ == "__main__":
    main()
