# PhotoPipe Rebuild Design

**Date:** 2026-05-17
**Status:** Draft for review
**Author:** PhotoPipe owner + Claude (Opus 4.7, 1M context)

---

## Why this exists

PhotoPipe today is a working but tightly-coupled Streamlit pipeline for digitizing family photos via the Epson FastFoto FF-680W. Two pressures drive a rebuild:

1. **A second pair of hands.** The owner wants to hand the running app to a helper (family member, hired assistant) who can scan stacks of photos without needing to know anything about batches, dates, locations, or AI configuration. The owner returns later to add context and run AI enrichment.
2. **The AI/OCR stack has materially improved since the original build.** Claude Sonnet 4.6 + prompt caching + Batch API + structured outputs offer a ~10–20× cost and quality jump over the current Sonnet 4 single-shot calls. Tesseract is now badly outclassed by frontier VLMs and purpose-built handwriting OCR services on the kind of faded ballpoint cursive that covers most photo backs.

The codebase audit found that the existing data model and core utilities (DB, scanner control, autocrop, metadata writer) are solid. The damage is in pages 2/3/4, which fuse business logic to the Streamlit render loop. The rebuild is **incremental, not a teardown**: keep ~70%, rewrite the page logic and AI client, add a Helper Mode surface and a phased data model.

---

## Goals

- **Helper Mode.** A scannerless-knowledge-required UI a non-owner can drive end-to-end: pick a bucket label, scan, see thumbnails, click done.
- **Two-phase data flow.** Photos live as `captured` → `curated` → `finalized`. Capture needs no owner context. Curate is owner-only and is where batch metadata + AI enrichment happens.
- **Modern LLM stack.** Sonnet 4.6 as default model. Prompt caching on the long instruction prefix. Batch API for >50-photo enrichment runs. Strict structured outputs (no more JSON parse failures). Multi-image per call (10–15 photos) for temporal coherence reasoning during curate.
- **Replace Tesseract with Mistral OCR 3** for handwriting on photo backs. Fall back to Sonnet 4.6 vision on low-confidence cases.
- **Extract pipelines from pages.** `capture_pipeline.py` and `curate_pipeline.py` are headless, callable from a Streamlit page, the CLI, or a test. The pages become thin renderers.
- **Survive macOS Tahoe Local Network drops.** Detect when `scanimage` discovery fails because of the permission landmine and surface an actionable error.

## Non-goals (this spec)

- Throwing out Streamlit. It works, the owner knows it, and the bottleneck is logic-in-pages.
- Throwing out the SQLite schema. Schema is clean; we add fields, not redesign.
- Multi-machine / multi-user concurrent capture. One scanner, one helper at a time.
- Cloud face recognition. (Face clustering is a deferred enhancement — see §10.)
- Local-only / on-device VLM mode. (Deferred — see §10.)
- A new scanner SDK (ImageCaptureCore / Swift). `scanimage` subprocess remains. Revisit only if SANE breaks.

---

## 1. Two-phase workflow

The current code conflates "scanned" with "ready for AI" with "ready for export." The rebuild makes the phase explicit on every photo and every UI surface.

```
                   ┌──────────────────┐
   helper / owner  │   CAPTURE PHASE  │   no batch, no AI, no owner context
                   │   (bucket label) │   just scan + OCR + autocrop + save
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
        owner only │   CURATE PHASE   │   convert bucket → batch
                   │   (add context)  │   add date range / location / people
                   │   (run AI)       │   multi-image AI coherence pass
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
        owner only │  FINALIZE PHASE  │   write EXIF/IPTC, copy to archive
                   │  (existing flow) │   generate report
                   └──────────────────┘
```

**Bucket** is a new lightweight concept: a free-text label the helper enters before scanning ("Grandma's blue album, page 3"). Buckets carry no metadata other than label + helper name + timestamp. The owner converts buckets to proper Batches during curate.

A photo always belongs to exactly one bucket and (after curate) one batch. The bucket-to-batch mapping can be many-to-one (owner merges several buckets) or one-to-many (owner splits a bucket by detected event segment).

## 2. Data model changes

Additive, no destructive migrations. All existing photos migrate to `phase = finalized` so the old workflow keeps working.

### New table: `buckets`
- `id` (uuid)
- `label` (text, helper-entered)
- `helper_name` (text, optional)
- `created_at` (timestamp)
- `batch_id` (uuid, nullable — set once owner converts to batch)
- `status` enum: `open` | `closed` | `converted`

### Changes to `photos` table
- Add `bucket_id` (uuid, nullable for back-compat)
- Add `phase` enum: `captured` | `curated` | `finalized` (default `finalized` for existing rows)
- Add `handwriting_ocr_text` (text) — separate column from existing `ocr_text_back`, since the new OCR is a different provider with different confidence semantics
- Add `handwriting_ocr_provider` enum: `mistral` | `claude_vlm` | `tesseract_legacy`
- Add `handwriting_ocr_confidence` (float 0–1)

### Changes to `batches` table
- Add `source_bucket_ids` (json array) — provenance for "this batch came from these buckets"

### New table: `ai_jobs`
For tracking Batch API jobs (which are async, can take hours):
- `id` (uuid)
- `batch_id` (uuid)
- `provider` (text) — `anthropic_batch`
- `provider_job_id` (text) — what we poll
- `status` enum: `queued` | `running` | `completed` | `failed`
- `submitted_at`, `completed_at`
- `photo_ids` (json array) — what's in this job
- `result_summary` (json, populated on completion)

---

## 3. Module structure

### Keep as-is
| Module | Why |
|---|---|
| `database.py` | Schema is clean. Adds covered by migrations. |
| `models.py` | Pydantic models extend cleanly. |
| `scanner.py` | `scanimage` wrapper is fine; gets new error-mapping for Tahoe Local Network. |
| `metadata.py` | ExifTool wrapper is stable. |
| `file_manager.py` | Output naming/copying is stable. |
| `geocoding.py` | Standalone utility, low risk. |
| `pairing.py` | Front/back matching logic still applies. |
| `config.py` | Add a few new sections (helper_mode, vlm), no restructure. |
| `autocrop.py` | Keep — but extract the embedded Anthropic orientation call into `vlm_client.py`. |

### Rewrite / extract
| Module | What happens |
|---|---|
| `pages/2_scan.py` | Becomes `pages/2_capture.py` — thin renderer over `capture_pipeline.py`. Stripped to helper-mode chrome. |
| `pages/3_review.py` | Becomes `pages/3_curate.py` — thin renderer over `curate_pipeline.py`. Multi-tab: Buckets → Context → AI → Review. |
| `pages/4_finalize.py` | Mostly unchanged. Extract export logic into `finalize_pipeline.py`. |
| `ai_dating.py` | Replaced by `vlm_client.py` (transport: caching, batch, structured output) + `dating_pipeline.py` (orchestration). |

### Net-new modules
| Module | Purpose |
|---|---|
| `capture_pipeline.py` | Headless: scan → crop → orient → pair → handwriting-OCR → save to bucket. Callable from page or CLI. |
| `curate_pipeline.py` | Headless: bucket→batch, AI orchestration, segment detection, apply results. |
| `finalize_pipeline.py` | Headless: write metadata, copy archive, generate report. |
| `vlm_client.py` | Single entry to Claude vision. Owns prompt caching (`cache_control` on prefix), Batch API submission/polling, structured output schema enforcement, retries. |
| `handwriting_ocr.py` | Mistral OCR 3 client. Crops back, sends in batches, parses confidence, falls back to `vlm_client.py` on low confidence. |
| `date_parser.py` | Extracted from current `ocr.py` — the regex patterns + month/season maps. Used by both Mistral and VLM paths. |
| `bucket_service.py` | CRUD + bucket→batch conversion. |
| `pages/0_buckets.py` | Owner-facing bucket dashboard: list buckets, see helper names, convert to batches. |
| `pages/2_capture.py` | Helper-mode page. Minimal chrome. |
| `pages/3_curate.py` | Owner curate page. |
| `cli/photopipe_capture` | CLI wrapper around capture_pipeline (for headless / scripted runs). |
| `cli/photopipe_curate` | CLI wrapper around curate_pipeline. |
| `cli/photopipe_doctor` | Diagnoses scanner discovery, Local Network permission, API keys, dependencies. |

### Delete
| Module | Why |
|---|---|
| `ocr.py` (Tesseract) | Replaced by `handwriting_ocr.py`. Keep `date_parser.py` extraction. |
| `pages/help.py` | Move content to README. Static help in a Streamlit page rots. |

---

## 4. Capture pipeline

A single function, callable headless:

```python
def capture_batch(
    bucket: Bucket,
    *,
    resolution: int = 600,
    duplex: bool = True,
    helper_name: Optional[str] = None,
    progress: Optional[Callable[[CaptureProgress], None]] = None,
) -> CaptureResult
```

Stages (each independently mockable for tests):
1. **Scan** — `scanner.scan_to_folder()` returns list of new file paths.
2. **Pair** — `pairing.pair_fronts_and_backs()` matches `_b` suffix or duplex order.
3. **Crop & orient** — `autocrop.process_scanned_photo()` per photo (existing logic).
4. **Handwriting OCR (async)** — submit all backs to Mistral OCR 3 as a single Batch API job. Don't block capture on this; queue results and poll. The bucket can show "OCR in progress" indicator.
5. **Date extraction** — when OCR returns, run `date_parser.parse_date_from_text()` and populate `photos.extracted_date` for any obvious matches (high-confidence regex on lab stamps, full-month-year strings).
6. **Persist** — write `PhotoPair` rows with `phase = captured`, `bucket_id = bucket.id`.

**No batch concept and no owner-context AI** during capture. The only AI calls in capture are mechanical, owner-context-free, and cheap:
- **Orientation detection** (existing): per-photo Claude call. Stays in capture — routed through `vlm_client.py` so it benefits from prompt caching across the session. Owner can disable in Settings if they want zero API spend during helper sessions; the system falls back to Apple Vision or naive heuristics.
- **Handwriting OCR**: Mistral OCR 3 (default) or Claude vision fallback. Submitted as a Batch API job so it doesn't block the helper and benefits from the 50% Batch discount.

The expensive owner-context-driven enrichment (period dating, location guess, evidence synthesis) happens in curate, not capture. A helper can scan all day with bounded, predictable per-photo cost.

## 5. Curate pipeline

Owner-driven. Composed of explicit stages the UI can call independently:

```python
def convert_bucket_to_batch(bucket: Bucket, batch_metadata: BatchMetadata) -> Batch: ...

def run_ai_dating(
    batch: Batch,
    photos: list[PhotoPair],
    *,
    mode: Literal["realtime", "batch_api"] = "realtime",
    images_per_call: int = 12,
) -> AIRunResult: ...

def apply_ai_results(
    batch: Batch,
    ai_run: AIRunResult,
    *,
    auto_apply_high_confidence: bool = False,
) -> ApplyResult: ...
```

### AI dating: multi-image strategy

The existing code sends 1 photo per call, then heuristically aggregates. The new approach sends 10–15 photos in a single Claude message with a structured-output schema asking for *per-photo* estimates plus a *coherence assessment* ("do these look like the same event/day/year? where does the timeline break?"). This:
- Cuts API calls 10–15×.
- Gives the model the context it needs to reason about temporal coherence (e.g., "the first 8 photos are clearly summer 1985 based on consistent fashion; photo 9 onward shows different people in winter clothing — likely a different event").
- Replaces the current janky `detect_segments` heuristic with model-driven segmentation.

### Realtime vs Batch API mode

- **Realtime** (default for <50 photos or interactive review): synchronous Claude call. Streamlit waits ~10–30s and shows a spinner.
- **Batch API** (default for ≥50 photos): submit job, store `ai_jobs` row, return immediately. Owner sees "AI running in background — check back in ~30 minutes." A small polling thread (or page-refresh check) updates `ai_jobs.status`. Cost: 50% off list, stacks with caching → ~95% off.

### Prompt caching

The instruction prompt (currently ~3k tokens) is identical across all photos in a session. Wrap it in `cache_control: { type: "ephemeral", ttl: "5m" }` (or `"1h"` for batch jobs). First call writes the cache at 1.25× input cost; every subsequent call in the TTL window pays 0.1× for the prefix. With 100+ photos per curate session, this is a 90% input-token discount on the bulk of the prompt.

### Structured output

Define a Pydantic schema for `BatchAIResult` and pass it via Anthropic's `output_config.format` or strict tool-use. Schema enforces the per-photo + per-segment structure. No more `try: json.loads(text)` with silent failures.

## 6. Handwriting OCR

```python
def ocr_photo_backs(
    photos: list[PhotoPair],
    *,
    mode: Literal["realtime", "batch_api"] = "batch_api",
) -> dict[PhotoId, OCRResult]
```

Pipeline per photo:
1. Crop back image to high-signal region (existing `autocrop` logic gives us an oriented back). Optionally pre-detect text region with a small text-detection model; skip in MVP.
2. Submit to Mistral OCR 3. Defaults to Batch API ($1/1K pages). Realtime mode ($2/1K) for single-photo interactive cases.
3. Parse confidence. Mistral returns per-region confidence; we average for a photo-level score.
4. If confidence < 0.6 OR no recognizable date pattern found, **fall back** to a Sonnet 4.6 vision call with a tighter prompt: "Read any handwritten or stamped text on this photo back. Return the literal text and any dates you can identify."
5. Run `date_parser.parse_date_from_text()` on the resulting text.
6. Persist `handwriting_ocr_text`, `handwriting_ocr_provider`, `handwriting_ocr_confidence`, and (if found) `extracted_date` + `date_source = ocr_back`.

Tesseract goes away. The legacy column `ocr_text_back` is preserved for existing photos but no longer written.

## 7. Helper Mode UI (`pages/2_capture.py`)

Design principle: a helper who has never seen the app before should be productive in 30 seconds and unable to break anything.

Layout (top to bottom):
- **Big header**: "📷 Scan Photos" — no app branding noise.
- **Bucket label input**: huge text field, placeholder "Where these came from (e.g., Grandma's blue album)". Pre-filled from last bucket if same session.
- **Your name** (optional, small): persisted to `localStorage` so it auto-fills next session.
- **One huge button**: "🟢 Scan Stack" — runs `capture_pipeline.capture_batch()`.
- **Live status**: scanning N of M, current photo, paper-jam detection.
- **Recent thumbnails**: scrolling grid of what just got scanned. No edit controls. No metadata. No batch selector. No settings.
- **Done button**: closes bucket, returns to bucket-label input.

Things explicitly NOT shown in helper mode:
- Batch creation/editing
- AI / dating controls
- Settings, API keys, configuration
- Sidebar, file paths, debug output
- Anything that could rack up an API bill if clicked

Helper-mode toggle lives in owner-side Settings: "Hand off to helper" → switches the app into helper mode (hides other pages, locks sidebar, simplifies chrome). Owner exits via a small "Owner login" link in the corner — guarded by a PIN set during setup. PIN is not security-grade — it's a "don't accidentally click into curate mode" guardrail.

## 8. Curate Mode UI (`pages/3_curate.py`)

Four tabs, walking the owner through the workflow:

1. **Buckets** — list of open + recently-converted buckets, with helper name, photo count, OCR completion %, thumbnail strip.
2. **Context** — for a selected bucket, fill in batch metadata: date range, location (geocoded), people (autocomplete from prior batches), event description. "Convert to Batch" button.
3. **AI** — for a converted batch, run AI dating. Choose realtime vs batch. Shows progress, evidence panel, segment view. Per-segment apply buttons (existing concept, but driven by structured AI output, not heuristics).
4. **Review** — thumbnail grid with multi-select, rotation, split-to-new-batch, approve/flag. Mostly preserves the current `3_review.py` thumbnail_grid UX since it works.

## 9. macOS Tahoe Local Network handling

Detection:
- On app start, run a quick scanner discovery probe (1-second timeout `scanimage -L`).
- If discovery returns no devices AND we expect a network scanner (config has `device: epsonds:net:...`), surface a clear banner: "Scanner not found. macOS may have dropped Local Network permission for this terminal — see [Troubleshoot]."
- The troubleshoot link opens a Streamlit dialog with: (a) how to grant permission (System Settings → Privacy → Local Network → check the entry for your terminal app or `python`), (b) note that Tahoe 26.3.1+ silently drops this periodically, (c) link to `photopipe doctor` CLI.

`photopipe doctor` CLI subcommand:
- Probes scanner discovery
- Verifies API keys (Anthropic, Mistral)
- Checks ExifTool, sane-backends versions
- Reports homebrew formula versions
- Suggests one specific fix per problem

## 10. Deferred enhancements (separate specs, not this rebuild)

These came up in research but are out of scope for the initial rebuild:

- **Face clustering with InspireFace.** Local, ANE-accelerated, no PII to cloud. Lets owner label one photo per cluster ("Grandma Rose") and propagate. Worth a separate ~1-week spec.
- **Local-only VLM mode (Qwen3.6 via mlx-vlm or Apple Foundation Models).** For privacy-conscious users or to eliminate per-photo cost. Quality is lower than Sonnet 4.6 — needs A/B testing on owner's actual photos first.
- **Multi-photo-per-scan splitter.** Research flagged this as a real market gap (commercial AutoSplitter exists, no good open-source). Could be a differentiator if owner ever scans multi-up.
- **Native ImageCaptureCore wrapper.** Only worth doing if SANE breaks on macOS 27. Apple's Image Capture is already unreliable on Tahoe.
- **Gemini 3.1 Pro for huge-batch coherence** (3000 images/request). Interesting at scale but adds a second provider; revisit only if Claude multi-image batching proves insufficient.

---

## 11. Migration plan (rough sequence)

1. **DB migration**: add new columns/tables with defaults that preserve existing behavior.
2. **Extract `date_parser.py`** from `ocr.py` (pure functions, no dependencies). Cover with tests.
3. **Build `vlm_client.py`** with prompt caching + structured output. Cover with mocked Anthropic client tests. Migrate `autocrop.py`'s embedded Anthropic call to use it.
4. **Build `handwriting_ocr.py`** with Mistral OCR 3 client + VLM fallback. Cover with mocked HTTP tests.
5. **Build `capture_pipeline.py`** by extracting and refactoring logic from `pages/2_scan.py`. Make it work headlessly via CLI first, then thread through to page.
6. **Build helper-mode `pages/2_capture.py`** as a thin renderer over the pipeline.
7. **Build `bucket_service.py` + `pages/0_buckets.py`** for the owner-side bucket dashboard.
8. **Build `curate_pipeline.py`** by extracting logic from `pages/3_review.py`. Replace heuristic segment detection with multi-image AI structured output.
9. **Build curate `pages/3_curate.py`** as a thin renderer.
10. **Add `photopipe doctor` CLI** + Tahoe Local Network detection.
11. **Delete `ocr.py`** after verifying no fallback paths still use it. Run a one-time migration to mark legacy OCR'd photos.
12. **Update README** to reflect helper-mode workflow, AI provider config, troubleshooting.

Each step ships behind the existing UI so the owner can keep using PhotoPipe while the rebuild proceeds.

## 12. Testing

The current `tests/` directory has only an `__init__.py`. The rebuild establishes a baseline:

- **Unit tests** for `date_parser`, `vlm_client` (mocked transport), `handwriting_ocr` (mocked transport), `bucket_service`, `capture_pipeline`, `curate_pipeline`.
- **Integration tests** for the full capture path (scanner mocked) and curate path (Anthropic API mocked with recorded responses).
- **Snapshot tests** for prompt construction (so a casual edit doesn't silently change cache-key behavior — caching is sensitive to byte-exact prefixes).
- **No UI tests for Streamlit pages.** Streamlit tests are flaky and the pages are now thin renderers; pipeline tests cover the logic.

Target: each new module ships with tests in the same PR/commit that introduces it.

---

## Open questions for the implementation plan

- **Helper-mode PIN strength.** Numeric 4-digit is fine for "don't accidentally click into curate." Anything stronger is theater unless we also encrypt the API key at rest. Defaults: 4-digit, owner can override.
- **Mistral OCR vs straight-to-VLM.** Mistral is cheaper and purpose-built but adds a second provider and a second API key. If the owner doesn't want a Mistral key, the system should gracefully fall back to Claude-only OCR (more expensive but simpler). Config flag: `ocr.provider: mistral | claude | auto`.
- **Bucket → batch UX when one bucket spans multiple events.** The AI segment detection runs after batch conversion, so the owner might convert a bucket to a batch, run AI, see segments, and want to retroactively split. The curate UI needs a "split this batch into N by AI-detected segments" action — already in scope per §8 but worth flagging.
- **Batch API job lifecycle.** What happens if the owner closes the app while a Batch API job is pending? The poller needs to be resumable on app restart. Plan: poll on first page load if any `ai_jobs.status IN ('queued', 'running')`.

These get resolved during planning, not here.
