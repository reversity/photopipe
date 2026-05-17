"""
Smoke test for the Mistral OCR 3 SDK response shape.

The HandwritingOCR module assumes:
  - mistral_client.ocr.process(model=..., document={"type":"image_url", ...})
  - response has `.pages` iterable
  - each page has `.markdown` (string)
  - pages may have `.regions` with `.confidence` floats

If the live SDK returns a different shape, _call_mistral in
photopipe/handwriting_ocr.py is the only place that needs to change.
This script makes one real Mistral OCR 3 call against a photo back you
provide, prints the response, and reports whether the assumed shape holds.

Usage:
    export MISTRAL_API_KEY=...
    python scripts/verify_mistral_ocr.py /path/to/photo_back.jpg
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a photo back image (.jpg)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: {args.image} does not exist", file=sys.stderr)
        return 2
    if not os.environ.get("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY environment variable is not set", file=sys.stderr)
        return 2

    print(f"Running Mistral OCR 3 on: {args.image}")
    print("-" * 60)

    try:
        from photopipe.handwriting_ocr import HandwritingOCR
        ocr = HandwritingOCR()
        result = ocr.ocr_back(args.image)
    except Exception as e:
        print(f"\n❌ Live call raised: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf this is an AttributeError on `.pages`, `.markdown`, or "
            "`.regions`, the SDK response shape has drifted. "
            "Edit `_call_mistral` in photopipe/handwriting_ocr.py.",
            file=sys.stderr,
        )
        return 1

    print(f"Provider: {result.provider}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Extracted date: {result.extracted_date}")
    print("Text:")
    print("-" * 60)
    print(result.text or "(empty)")
    print("-" * 60)

    if result.provider == "mistral":
        print("\n✅ Mistral path succeeded — assumed SDK shape holds.")
    elif result.provider == "claude":
        print(
            "\n⚠️  Fell back to Claude VLM (Mistral confidence below threshold "
            "or Mistral path raised). The Claude path works regardless. "
            "Inspect Mistral logs/credentials if you expected the Mistral path."
        )
    else:
        print(f"\n❓ Unexpected provider: {result.provider}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
