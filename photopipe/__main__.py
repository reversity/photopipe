"""
PhotoPipe CLI - Command-line interface for batch operations.

Usage:
    python -m photopipe init
    python -m photopipe batch create --name "Summer_1985" --date-start 1985-06-01
    python -m photopipe batch process --name "Summer_1985"
    python -m photopipe batch finalize --name "Summer_1985"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from photopipe import __version__
from photopipe.config import get_config, save_config
from photopipe.database import Database
from photopipe.models import Batch, BatchStatus
from photopipe.pairing import pair_scans, get_pairing_summary
from photopipe.geocoding import geocode_location
from photopipe.file_manager import finalize_batch
from photopipe.curate_pipeline import run_ai_dating, apply_ai_results
from photopipe.handwriting_ocr import HandwritingOCR
from photopipe.models import DateSource


def cmd_init(args):
    """Initialize PhotoPipe configuration and database."""
    print("Initializing PhotoPipe...")

    config = get_config()
    config.ensure_directories()

    print(f"  Created input folder: {config.paths.input_folder}")
    print(f"  Created output folder: {config.paths.output_folder}")
    print(f"  Created archive folder: {config.paths.archive_folder}")

    # Initialize database
    db = Database()
    print(f"  Initialized database: {config.paths.database}")

    # Save config if not exists
    config_path = Path.home() / ".photopipe" / "config.yaml"
    if not config_path.exists():
        save_config(config, config_path)
        print(f"  Created config file: {config_path}")

    print("\n✅ PhotoPipe initialized successfully!")
    print("\nNext steps:")
    print("  1. (Optional) Set ANTHROPIC_API_KEY for AI dating")
    print("  2. Run: streamlit run app.py")


def cmd_batch_list(args):
    """List all batches."""
    db = Database()
    batches = db.get_all_batches()

    if not batches:
        print("No batches found.")
        return

    print(f"\n{'Name':<30} {'Status':<12} {'Photos':<8} {'Date Range'}")
    print("-" * 80)

    for batch in batches:
        stats = db.get_batch_stats(batch.id)
        print(f"{batch.name:<30} {batch.status:<12} {stats['total']:<8} {batch.get_date_range_str()}")


def cmd_batch_create(args):
    """Create a new batch."""
    db = Database()
    config = get_config()

    # Check if batch exists
    existing = db.get_batch_by_name(args.name)
    if existing:
        print(f"Error: Batch '{args.name}' already exists.")
        sys.exit(1)

    # Parse dates
    date_start = None
    date_end = None
    if args.date_start:
        date_start = datetime.strptime(args.date_start, "%Y-%m-%d").date()
    if args.date_end:
        date_end = datetime.strptime(args.date_end, "%Y-%m-%d").date()

    # Geocode location if provided
    location = None
    if args.location:
        print(f"Geocoding location: {args.location}")
        location = geocode_location(args.location)
        if location:
            print(f"  Found: {location.address}")
        else:
            print("  Warning: Could not geocode location")

    # Parse people
    people = []
    if args.people:
        people = [p.strip() for p in args.people.split(",")]

    # Create batch
    batch = Batch(
        name=args.name,
        date_start=date_start,
        date_end=date_end,
        location_description=args.location,
        location=location,
        event_description=args.description,
        people=people,
        input_folder=Path(args.input_folder) if args.input_folder else config.paths.input_folder,
    )

    db.create_batch(batch)
    print(f"\n✅ Created batch: {args.name}")
    print(f"   ID: {batch.id}")


def cmd_batch_process(args):
    """Ingest pairs, run handwriting OCR on backs, optionally run AI dating.

    Drives the new rebuild pipeline headlessly:
      - pair_scans ingests fronts/backs from the batch input folder
      - HandwritingOCR (Mistral OCR 3 → Claude fallback) runs on each back
      - --ai-dating triggers curate_pipeline.run_ai_dating for multi-image
        period/location reasoning

    Use `--preview` to inspect pairs without writing anything.
    """
    db = Database()

    batch = db.get_batch_by_name(args.name)
    if not batch:
        print(f"Error: Batch '{args.name}' not found.")
        sys.exit(1)

    input_folder = batch.input_folder or get_config().paths.input_folder

    # Preview
    if args.preview:
        summary = get_pairing_summary(input_folder)
        print(f"\nInput folder: {input_folder}")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Complete pairs: {summary['complete_pairs']}")
        print(f"  Fronts without backs: {summary['fronts_without_backs']}")
        print(f"  Orphaned backs: {summary['orphaned_backs']}")
        return

    # Step 1: Ingest pairs into DB
    print(f"\n📥 Ingesting photos from {input_folder}...")
    new_photos = pair_scans(input_folder, batch, db)
    print(f"   Ingested {len(new_photos)} new photo pairs")

    if not new_photos:
        print("   No new photos to process")
        return

    # Step 2: Handwriting OCR on backs (Mistral OCR 3 → Claude fallback)
    if not args.skip_ocr:
        backs = [p for p in new_photos if p.back_path is not None]
        if backs:
            print(f"\n📝 Running handwriting OCR on {len(backs)} photo backs...")
            ocr = HandwritingOCR()
            for i, photo in enumerate(backs, 1):
                try:
                    result = ocr.ocr_back(photo.back_path)
                    photo.handwriting_ocr_text = result.text
                    photo.handwriting_ocr_provider = result.provider
                    photo.handwriting_ocr_confidence = result.confidence
                    if result.extracted_date:
                        photo.extracted_date = result.extracted_date
                        photo.date_source = DateSource.OCR_BACK
                    db.update_photo(photo)
                    print(f"   [{i}/{len(backs)}] {photo.front_path.name}: "
                          f"{result.provider} conf={result.confidence:.2f}")
                except Exception as e:
                    print(f"   [{i}/{len(backs)}] {photo.front_path.name}: OCR failed: {e}")

    # Step 3: Multi-image AI dating (curate_pipeline)
    if args.ai_dating:
        undated = [p for p in new_photos if not p.extracted_date]
        if undated:
            print(f"\n🤖 Running AI dating on {len(undated)} undated photos "
                  f"(multi-image, 12 photos per call)...")
            result = run_ai_dating(batch, undated, images_per_call=12)
            print(f"   {len(result.raw_responses)} AI call(s)")
            if result.coherence.get("segment_breaks"):
                print(f"   AI detected {len(result.coherence['segment_breaks'])} "
                      f"segment break(s) — review in the curate UI")
            applied = apply_ai_results(batch, result, undated, db=db)
            print(f"   Applied dates to {applied.updated} photos "
                  f"(skipped {applied.skipped})")

    batch.status = BatchStatus.PROCESSING
    db.update_batch(batch)

    print(f"\n✅ Process complete!")
    print(f"   Next: `streamlit run app.py` to curate, then finalize.")


def cmd_batch_finalize(args):
    """Finalize a batch and export."""
    db = Database()

    batch = db.get_batch_by_name(args.name)
    if not batch:
        print(f"Error: Batch '{args.name}' not found.")
        sys.exit(1)

    stats = db.get_batch_stats(batch.id)
    print(f"\n📊 Batch: {batch.name}")
    print(f"   Total photos: {stats['total']}")
    print(f"   Needs review: {stats['needs_review']}")

    if stats['needs_review'] > 0 and not args.auto_approve:
        print("\n⚠️  Some photos need review.")
        print("   Use --auto-approve-high-confidence to auto-approve high confidence dates")
        print("   Or review in the web interface first")
        if not args.force:
            sys.exit(1)

    print("\n✨ Finalizing batch...")

    def progress(current, total):
        print(f"   Finalizing {current}/{total}...", end="\r")

    report = finalize_batch(
        batch,
        db,
        auto_approve_high_confidence=args.auto_approve,
        progress_callback=progress,
    )

    print(f"\n✅ Finalization complete!")
    print(f"   Photos processed: {report.photo_count}")
    print(f"   Output location: {get_config().paths.output_folder}")


def cmd_batch_delete(args):
    """Delete a batch."""
    db = Database()

    batch = db.get_batch_by_name(args.name)
    if not batch:
        print(f"Error: Batch '{args.name}' not found.")
        sys.exit(1)

    if not args.force:
        confirm = input(f"Delete batch '{args.name}'? This cannot be undone. [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    db.delete_batch(batch.id)
    print(f"✅ Deleted batch: {args.name}")


def cmd_doctor(args):
    """Diagnose PhotoPipe environment and exit non-zero on failure."""
    from photopipe.cli.doctor import run_doctor
    sys.exit(run_doctor())


def cmd_status(args):
    """Show system status."""
    config = get_config()
    db = Database()

    print("\n📷 PhotoPipe Status")
    print("=" * 50)

    # Paths
    print("\n📁 Paths:")
    print(f"   Input folder:  {config.paths.input_folder}")
    print(f"   Output folder: {config.paths.output_folder}")
    print(f"   Archive:       {config.paths.archive_folder}")
    print(f"   Database:      {config.paths.database}")

    # Dependencies
    print("\n🔧 Dependencies:")

    from photopipe.metadata import check_exiftool_installed

    exiftool = "✅" if check_exiftool_installed() else "❌"

    print(f"   ExifTool:  {exiftool}")

    # AI
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    ai_status = "✅" if api_key else "⚠️ No API key"
    print(f"   AI Dating: {ai_status}")

    # Batches
    batches = db.get_all_batches()
    print(f"\n📊 Batches: {len(batches)}")

    pending = len([b for b in batches if b.status == "pending"])
    review = len([b for b in batches if b.status == "review"])
    complete = len([b for b in batches if b.status == "complete"])

    print(f"   Pending:  {pending}")
    print(f"   Review:   {review}")
    print(f"   Complete: {complete}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="photopipe",
        description="PhotoPipe - Photo Scanning Metadata Pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize PhotoPipe")
    init_parser.set_defaults(func=cmd_init)

    # status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor", help="Diagnose common setup issues (deps, scanner, API keys)"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # batch command group
    batch_parser = subparsers.add_parser("batch", help="Batch management commands")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command")

    # batch list
    batch_list_parser = batch_subparsers.add_parser("list", help="List all batches")
    batch_list_parser.set_defaults(func=cmd_batch_list)

    # batch create
    batch_create_parser = batch_subparsers.add_parser("create", help="Create a new batch")
    batch_create_parser.add_argument("--name", required=True, help="Batch name")
    batch_create_parser.add_argument("--date-start", help="Start date (YYYY-MM-DD)")
    batch_create_parser.add_argument("--date-end", help="End date (YYYY-MM-DD)")
    batch_create_parser.add_argument("--location", help="Location description")
    batch_create_parser.add_argument("--description", help="Event description")
    batch_create_parser.add_argument("--people", help="Comma-separated list of people")
    batch_create_parser.add_argument("--input-folder", help="Scanner input folder")
    batch_create_parser.set_defaults(func=cmd_batch_create)

    # batch process
    batch_process_parser = batch_subparsers.add_parser("process", help="Process a batch")
    batch_process_parser.add_argument("--name", required=True, help="Batch name")
    batch_process_parser.add_argument("--preview", action="store_true", help="Preview only")
    batch_process_parser.add_argument("--ai-dating", action="store_true", help="Run multi-image AI dating after OCR")
    batch_process_parser.add_argument("--skip-ocr", action="store_true", help="Skip handwriting OCR on backs")
    batch_process_parser.set_defaults(func=cmd_batch_process)

    # batch finalize
    batch_finalize_parser = batch_subparsers.add_parser("finalize", help="Finalize a batch")
    batch_finalize_parser.add_argument("--name", required=True, help="Batch name")
    batch_finalize_parser.add_argument(
        "--auto-approve-high-confidence",
        dest="auto_approve",
        action="store_true",
        help="Auto-approve high confidence dates",
    )
    batch_finalize_parser.add_argument("--force", action="store_true", help="Force finalize even with review needed")
    batch_finalize_parser.set_defaults(func=cmd_batch_finalize)

    # batch delete
    batch_delete_parser = batch_subparsers.add_parser("delete", help="Delete a batch")
    batch_delete_parser.add_argument("--name", required=True, help="Batch name")
    batch_delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    batch_delete_parser.set_defaults(func=cmd_batch_delete)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "batch" and not args.batch_command:
        batch_parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
