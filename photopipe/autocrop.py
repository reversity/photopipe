"""
Auto-crop module for detecting and extracting photos from full-page scans.

The Epson FastFoto scans full pages - this module detects the actual photo
within the scan and crops to just that region.

Also includes AI-based orientation detection to auto-rotate photos correctly.
"""

import cv2
import numpy as np
import base64
import io
import os
import json
import re
import shutil
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


def _normalize_orientation(exif_bytes: Optional[bytes]) -> Optional[bytes]:
    """Drop the EXIF Orientation tag from an EXIF block.

    ``cv2.imread`` physically applies the Orientation tag when it decodes, so
    the pixels we re-save are already display-correct. Re-embedding the
    original Orientation (e.g. 6) would tell viewers to rotate a second time.
    Strip it so the saved crop is oriented exactly once.
    """
    if not exif_bytes:
        return exif_bytes
    try:
        from PIL import Image as _I

        probe = _I.new("RGB", (1, 1))
        exif = probe.getexif()
        exif.load(exif_bytes)
        if 274 in exif:  # 0x0112 Orientation
            del exif[274]
            return exif.tobytes()
        return exif_bytes
    except Exception:
        return exif_bytes


def _read_image_meta(path: Path) -> tuple[Optional[bytes], Optional[tuple]]:
    """Read the EXIF block and DPI from an image so a re-save can preserve both.

    Scanner JPEGs store the scan resolution in the JFIF density header (PIL's
    ``info['dpi']``), NOT in EXIF — so preserving EXIF alone drops the DPI and
    a re-saved crop reports as 1 pixel/inch. We carry the DPI explicitly, and
    strip the Orientation tag (cv2 already applied it) to avoid double-rotation.
    """
    try:
        with Image.open(path) as img:
            exif = _normalize_orientation(img.info.get("exif"))
            dpi = img.info.get("dpi")
            # PIL reports (1, 1) when the source had no real density; treat
            # that as "unknown" so we can fall back to a sane default.
            if dpi and (dpi[0] or 0) > 1 and (dpi[1] or 0) > 1:
                return exif, (int(round(dpi[0])), int(round(dpi[1])))
            return exif, None
    except Exception:
        return None, None


def _save_bgr_jpeg(
    image: np.ndarray,
    output_path: Path,
    exif: Optional[bytes],
    dpi: Optional[tuple],
) -> None:
    """Save an OpenCV BGR array as JPEG, carrying over source EXIF and DPI."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    kwargs = {"quality": 95}
    if exif:
        kwargs["exif"] = exif
    if dpi:
        kwargs["dpi"] = dpi
    pil_img.save(output_path, "JPEG", **kwargs)


def get_anthropic_api_key() -> Optional[str]:
    """Get the Anthropic API key from settings or environment."""
    # First try environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    # Then try settings file
    settings_path = Path.home() / ".photopipe" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            api_key = settings.get("anthropic_api_key")
            if api_key:
                return api_key
        except Exception:
            pass

    return None


# Standard photo aspect ratios (long side / short side)
# These are the most common printed photo sizes
STANDARD_PHOTO_RATIOS = [
    (1.0, "square"),        # Polaroid, Instagram prints
    (1.25, "8x10"),         # 8x10 prints
    (1.33, "4x5"),          # 4x5 prints (close to 4:3)
    (1.4, "5x7"),           # 5x7 prints
    (1.43, "3.5x5"),        # 3.5x5 prints
    (1.5, "4x6"),           # 4x6 prints (most common, 2:3)
    (1.667, "3x5"),         # 3x5 prints (older format)
]

# Tolerance for matching ratios (as a fraction of the ratio)
# 5% tolerance catches slight variations while rejecting truly wrong crops
RATIO_TOLERANCE = 0.05


def find_closest_standard_ratio(width: int, height: int) -> Tuple[float, str]:
    """
    Find the closest standard photo ratio for given dimensions.

    Args:
        width: Image width
        height: Image height

    Returns:
        Tuple of (ratio, name) for the closest standard ratio
    """
    # Calculate aspect ratio (always as larger/smaller)
    if width >= height:
        current_ratio = width / height
    else:
        current_ratio = height / width

    # Find the closest standard ratio
    best_ratio = STANDARD_PHOTO_RATIOS[0]
    best_diff = abs(current_ratio - best_ratio[0])

    for ratio, name in STANDARD_PHOTO_RATIOS:
        diff = abs(current_ratio - ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio = (ratio, name)

    return best_ratio


def is_standard_ratio(width: int, height: int) -> bool:
    """
    Check if dimensions match a standard photo ratio.

    Args:
        width: Image width
        height: Image height

    Returns:
        True if within tolerance of a standard ratio
    """
    if width >= height:
        current_ratio = width / height
    else:
        current_ratio = height / width

    for ratio, _ in STANDARD_PHOTO_RATIOS:
        if abs(current_ratio - ratio) / ratio <= RATIO_TOLERANCE:
            return True

    return False


def adjust_crop_to_standard_ratio(
    x: int, y: int, w: int, h: int,
    image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    """
    Adjust crop region to match the closest standard photo ratio.

    Expands the crop slightly if possible, otherwise contracts it,
    to achieve a standard aspect ratio while keeping the center point.

    Args:
        x, y, w, h: Current crop region
        image_width, image_height: Full image dimensions

    Returns:
        Adjusted (x, y, w, h) tuple
    """
    # Already standard? Keep it
    if is_standard_ratio(w, h):
        return (x, y, w, h)

    # Find target ratio
    target_ratio, ratio_name = find_closest_standard_ratio(w, h)

    # Determine orientation
    is_landscape = w >= h

    # Calculate center of current crop
    center_x = x + w // 2
    center_y = y + h // 2

    if is_landscape:
        # Landscape: width is the long side
        # Try to keep width, adjust height
        target_h = int(w / target_ratio)
        target_w = w

        # If target height is taller than we have, adjust width instead
        if target_h > h:
            # Need more height than available - shrink width to match height
            target_w = int(h * target_ratio)
            target_h = h
    else:
        # Portrait: height is the long side
        # Try to keep height, adjust width
        target_w = int(h / target_ratio)
        target_h = h

        # If target width is wider than we have, adjust height instead
        if target_w > w:
            # Need more width than available - shrink height to match width
            target_h = int(w * target_ratio)
            target_w = w

    # Calculate new position to keep center
    new_x = center_x - target_w // 2
    new_y = center_y - target_h // 2

    # Clamp to image bounds
    new_x = max(0, min(new_x, image_width - target_w))
    new_y = max(0, min(new_y, image_height - target_h))

    # Final bounds check
    if new_x + target_w > image_width:
        target_w = image_width - new_x
    if new_y + target_h > image_height:
        target_h = image_height - new_y

    return (new_x, new_y, target_w, target_h)


def detect_photo_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the photo region within a full-page scan.

    Args:
        image: OpenCV image (BGR format)

    Returns:
        Tuple of (x, y, width, height) or None if no photo detected
    """
    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to find edges
    # This works well for photos on white/light backgrounds
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Also try edge detection for better results
    edges = cv2.Canny(blurred, 30, 100)

    # Combine threshold and edges
    combined = cv2.bitwise_or(thresh, edges)

    # Dilate to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour that's roughly rectangular and not the full page
    best_rect = None
    best_area = 0
    page_area = width * height
    min_area = page_area * 0.05  # At least 5% of page
    max_area = page_area * 0.85  # No more than 85% of page

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Filter by size
        if area < min_area or area > max_area:
            continue

        # Check aspect ratio (photos are typically 3:2 to 5:4, Polaroids are more square)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        if aspect > 3:  # Too elongated
            continue

        # Prefer larger areas
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    return best_rect


def detect_photo_by_color(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect photo region by looking for non-white areas.

    This is a fallback method that looks for the region that isn't
    pure white (the scanner background).

    Args:
        image: OpenCV image (BGR format)

    Returns:
        Tuple of (x, y, width, height) or None if no photo detected
    """
    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find pixels that aren't pure white (< 250)
    # The scanner background is typically pure white (255)
    mask = gray < 250

    # Find the bounding box of non-white pixels
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add small padding
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)

    w = x_max - x_min
    h = y_max - y_min

    # Sanity check - should be at least 10% of the page
    if w * h < (width * height * 0.1):
        return None

    return (x_min, y_min, w, h)


def _scan_dpi() -> Optional[tuple]:
    """The configured scan resolution as a DPI tuple (fallback for saves)."""
    try:
        from photopipe.config import get_config

        res = int(get_config().scanner.resolution)
        return (res, res)
    except Exception:
        return None


def _find_photo_contour(image: np.ndarray):
    """Find the outer contour of the photo on the (near-white) scanner bed.

    Returns the largest non-background contour, or None. Works on a raw
    full-page scan; unreliable on an already-cropped image (the photo fills
    the frame), which is why deskew must run from the pristine original.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    # Close gaps within the photo, then drop small speckle from dust/edges.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    page_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(c) < page_area * 0.02:
        return None
    return c


def _normalized_min_area_rect(contour):
    """minAreaRect with the angle mapped to the smallest straightening rotation.

    Returns (center, (w, h), angle) with angle in [-45, 45], or None.
    """
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < 20 or rh < 20:
        return None
    if angle < -45:
        angle += 90
        rw, rh = rh, rw
    elif angle > 45:
        angle -= 90
        rw, rh = rh, rw
    return (cx, cy), (rw, rh), angle


def deskew_and_crop(image: np.ndarray, max_angle: float = 20.0):
    """Straighten a tilted photo and crop it out of a full-page scan.

    Rotates the image so the photo is upright, then extracts it — removing the
    tilt and the white triangular corners a plain axis-aligned crop leaves on a
    skewed photo. Returns the cropped BGR array, or None if no photo found.
    """
    contour = _find_photo_contour(image)
    if contour is None:
        return None
    norm = _normalized_min_area_rect(contour)
    if norm is None:
        return None
    (cx, cy), (rw, rh), angle = norm
    if abs(angle) > max_angle:  # large angle => bad detection, don't rotate
        angle = 0.0
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )
    # Trim a hair so any sub-pixel white edge from the rotation is gone.
    size = (max(1, int(round(rw)) - 4), max(1, int(round(rh)) - 4))
    return cv2.getRectSubPix(rotated, size, (cx, cy))


def auto_crop_photo(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Auto-crop and deskew a scanned image to just the photo region.

    Args:
        input_path: Path to the scanned image
        output_path: Path to save cropped image (defaults to overwriting input)

    Returns:
        True if cropping was successful, False otherwise
    """
    if output_path is None:
        output_path = input_path

    try:
        # Load image with OpenCV
        image = cv2.imread(str(input_path))
        if image is None:
            return False

        height, width = image.shape[:2]

        # If the photo was fed with a meaningful tilt, deskew (straighten) it —
        # a plain axis-aligned crop would leave it tilted with white corners.
        # Straight photos skip this and use the proven tight crop below.
        contour = _find_photo_contour(image)
        if contour is not None:
            norm = _normalized_min_area_rect(contour)
            if norm is not None and abs(norm[2]) >= 1.5:
                deskewed = deskew_and_crop(image)
                if deskewed is not None and deskewed.size > 0:
                    deskewed = trim_scanner_border(deskewed)
                    exif, dpi = _read_image_meta(input_path)
                    _save_bgr_jpeg(deskewed, output_path, exif, dpi or _scan_dpi())
                    return True

        # Straight photo (or deskew not applicable): axis-aligned detection.
        rect = detect_photo_region(image)

        # Fall back to color-based detection
        if rect is None:
            rect = detect_photo_by_color(image)

        if rect is None:
            # No photo detected, keep original
            return False

        x, y, w, h = rect

        # Adjust crop to standard photo ratio before trimming borders
        # This ensures we're targeting a real photo size
        x, y, w, h = adjust_crop_to_standard_ratio(x, y, w, h, width, height)

        # Add additional margin trimming to remove scanner bed border
        # The scanner bed is light blue/gray - trim any border that's close to that color
        cropped = image[y:y+h, x:x+w]

        # Trim light blue/gray borders more aggressively
        cropped = trim_scanner_border(cropped)

        # After border trimming, adjust again to standard ratio
        crop_h, crop_w = cropped.shape[:2]
        if not is_standard_ratio(crop_w, crop_h):
            # Re-adjust to standard ratio
            _, _, new_w, new_h = adjust_crop_to_standard_ratio(0, 0, crop_w, crop_h, crop_w, crop_h)
            # Center the final crop
            trim_x = (crop_w - new_w) // 2
            trim_y = (crop_h - new_h) // 2
            cropped = cropped[trim_y:trim_y+new_h, trim_x:trim_x+new_w]

        # Save the cropped image, preserving the scan's EXIF + DPI metadata.
        # DPI defaults to the configured scan resolution when the source JPEG
        # carried no usable density (so a crop never reports as 1 px/inch).
        exif, dpi = _read_image_meta(input_path)
        if dpi is None:
            try:
                from photopipe.config import get_config

                res = int(get_config().scanner.resolution)
                dpi = (res, res)
            except Exception:
                dpi = None
        _save_bgr_jpeg(cropped, output_path, exif, dpi)

        return True

    except Exception as e:
        print(f"Auto-crop error: {e}")
        return False


def trim_scanner_border(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Trim the light blue/gray scanner bed border from an image.

    The scanner bed is approximately RGB (200, 210, 220) - a light blue-gray.
    This function trims rows/columns from edges that match this color.

    Args:
        image: BGR image array
        threshold: How many pixels from edge to check

    Returns:
        Trimmed image
    """
    height, width = image.shape[:2]

    # Scanner bed color detection - more permissive to catch variations
    def is_scanner_border_color(pixel_mean):
        """Check if a mean color matches scanner bed (light blue-gray)."""
        if len(pixel_mean) >= 3:
            b, g, r = pixel_mean[:3]
            # Scanner bed is light gray-blue: high values, relatively uniform, slight blue tint
            # Also catch pure light gray areas
            is_light = b > 170 and g > 170 and r > 170
            is_uniform = abs(b - g) < 40 and abs(g - r) < 40
            return is_light and is_uniform
        return False

    def is_scanner_border_row(row):
        """Check if a row is mostly scanner bed color."""
        mean_color = np.mean(row, axis=0)
        return is_scanner_border_color(mean_color)

    def is_scanner_border_col(col):
        """Check if a column is mostly scanner bed color."""
        mean_color = np.mean(col, axis=0)
        return is_scanner_border_color(mean_color)

    def find_content_start(check_func, start, end, step):
        """Find where actual content starts by looking for sustained non-border region."""
        consecutive_content = 0
        required_consecutive = 5  # Need 5 consecutive non-border rows/cols
        last_border = start

        for i in range(start, end, step):
            if check_func(i):
                # This is a border row/col
                consecutive_content = 0
                last_border = i
            else:
                consecutive_content += 1
                if consecutive_content >= required_consecutive:
                    # Found sustained content, return position just past last border
                    return last_border + step if step > 0 else last_border + step
        return start

    # Find borders with sustained content detection
    # Search up to 40% of image dimension or 1500px, whichever is larger
    max_border_search = 1500

    # Top border - scan down looking for 5 consecutive content rows
    top = 0
    consecutive_content = 0
    for i in range(min(height * 2 // 5, max_border_search)):
        if is_scanner_border_row(image[i]):
            consecutive_content = 0
            top = i + 1
        else:
            consecutive_content += 1
            if consecutive_content >= 5:
                break

    # Bottom border - scan up looking for 5 consecutive content rows
    bottom = height
    consecutive_content = 0
    for i in range(height - 1, max(height * 3 // 5, height - max_border_search), -1):
        if is_scanner_border_row(image[i]):
            consecutive_content = 0
            bottom = i
        else:
            consecutive_content += 1
            if consecutive_content >= 5:
                break

    # Left border - scan right looking for 5 consecutive content cols
    left = 0
    consecutive_content = 0
    for i in range(min(width * 2 // 5, max_border_search)):
        if is_scanner_border_col(image[:, i]):
            consecutive_content = 0
            left = i + 1
        else:
            consecutive_content += 1
            if consecutive_content >= 5:
                break

    # Right border - scan left looking for 5 consecutive content cols
    right = width
    consecutive_content = 0
    for i in range(width - 1, max(width * 3 // 5, width - max_border_search), -1):
        if is_scanner_border_col(image[:, i]):
            consecutive_content = 0
            right = i
        else:
            consecutive_content += 1
            if consecutive_content >= 5:
                break

    # Ensure we have a valid crop region (at least 50% of original in each dimension)
    if right <= left or bottom <= top:
        return image
    if (right - left) < width * 0.5 or (bottom - top) < height * 0.5:
        return image  # Cropped too much, return original

    # Small margin to avoid cutting into photo
    margin = 3
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(height, bottom + margin)
    right = min(width, right + margin)

    return image[top:bottom, left:right]


def auto_rotate_photo(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Auto-rotate is disabled - photos are fed in different orientations.

    Use rotate_photo() for manual rotation via UI controls.

    Args:
        input_path: Path to the image
        output_path: Path to save rotated image (defaults to overwriting input)

    Returns:
        False (no auto-rotation performed)
    """
    # Auto-rotation disabled - photos have different orientations
    # User should use manual rotation buttons in the UI
    return False


def rotate_photo(input_path: Path, degrees: int) -> bool:
    """
    Rotate a photo by specified degrees.

    Args:
        input_path: Path to the image
        degrees: Rotation in degrees (90, 180, 270, -90, etc.)

    Returns:
        True if successful
    """
    try:
        with Image.open(input_path) as img:
            # Strip Orientation: we're rotating the pixels ourselves, so a
            # leftover tag would make viewers rotate a second time.
            exif = _normalize_orientation(img.info.get("exif"))
            dpi = img.info.get("dpi")
            rotated = img.rotate(degrees, expand=True)
        kwargs = {"quality": 95}
        if exif:
            kwargs["exif"] = exif
        # Preserve DPI across the rotate re-save (PIL otherwise defaults it to 1)
        if dpi and (dpi[0] or 0) > 1 and (dpi[1] or 0) > 1:
            kwargs["dpi"] = (int(round(dpi[0])), int(round(dpi[1])))
        rotated.save(input_path, **kwargs)
        return True
    except Exception as e:
        print(f"Rotate error: {e}")
        return False


ORIENTATION_PROMPT = """Look at this photo. Which edge of the image currently contains what should be at the TOP when correctly oriented? (Clues: people's heads, sky or ceiling, the tops of buildings/trees, readable text.)

Reply with exactly one word and nothing else:
TOP (already correct) | BOTTOM (upside down) | LEFT (left edge should be on top) | RIGHT (right edge should be on top)"""


def detect_orientation_with_ai(image_path: Path) -> int:
    """
    Use Claude Vision to detect the correct orientation of a photo.

    Args:
        image_path: Path to the photo

    Returns:
        Rotation needed in degrees (0, 90, 180, or 270)
        Note: PIL rotates counter-clockwise, so:
        - 90 = counter-clockwise (what we call rotate_left)
        - 270 = clockwise (what we call rotate_right)
    """
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed")
        return 0

    api_key = get_anthropic_api_key()
    if not api_key:
        print("No Anthropic API key found")
        return 0

    try:
        # Load and resize image for API
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Use larger size for better orientation detection
        max_dim = 1024
        width, height = img.size
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        # Call Claude
        from photopipe.config import get_config

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=get_config().vlm.model,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": ORIENTATION_PROMPT,
                        },
                    ],
                }
            ],
        )

        result = response.content[0].text.strip().upper()
        print(f"AI orientation detection for {image_path.name}: {result}")

        # Parse the response - which edge should be TOP?
        # Match whole words and take the LAST one, so any preamble that
        # happens to mention an edge word can't shadow the actual answer.
        # PIL rotate() rotates counter-clockwise by default; we rotate so
        # the named edge becomes the new top.
        words = re.findall(r"\b(TOP|BOTTOM|LEFT|RIGHT)\b", result)
        answer = words[-1] if words else None

        if answer == "TOP":
            # Already correct
            return 0
        elif answer == "BOTTOM":
            # Upside down - rotate 180
            return 180
        elif answer == "LEFT":
            # Left edge should become top - rotate clockwise (PIL: negative)
            return -90
        elif answer == "RIGHT":
            # Right edge should become top - rotate counter-clockwise (PIL: positive)
            return 90
        else:
            print(f"Unexpected orientation response: {result}")
            return 0

    except Exception as e:
        print(f"AI orientation detection error: {e}")
        return 0


def auto_orient_photo(input_path: Path) -> int:
    """
    Automatically orient a photo using AI detection.

    Args:
        input_path: Path to the photo

    Returns:
        Degrees rotated (0 if no rotation needed)
    """
    rotation = detect_orientation_with_ai(input_path)

    if rotation != 0:
        rotate_photo(input_path, rotation)

    return rotation


def get_photo_ratio_info(image_path: Path) -> dict:
    """
    Get aspect ratio information for a photo.

    Args:
        image_path: Path to the photo

    Returns:
        Dict with ratio information
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        ratio, ratio_name = find_closest_standard_ratio(width, height)
        is_standard = is_standard_ratio(width, height)

        if width >= height:
            actual_ratio = width / height
        else:
            actual_ratio = height / width

        return {
            "width": width,
            "height": height,
            "actual_ratio": round(actual_ratio, 3),
            "closest_standard": ratio_name,
            "standard_ratio": ratio,
            "is_standard": is_standard,
        }
    except Exception as e:
        return {"error": str(e)}


def process_scanned_photo(input_path: Path, output_path: Optional[Path] = None, use_ai_orientation: bool = True) -> dict:
    """
    Process a scanned photo: auto-crop and auto-orient using AI.

    Args:
        input_path: Path to the scanned image
        output_path: Path to save processed image (defaults to overwriting input)
        use_ai_orientation: Whether to use AI to detect and correct orientation

    Returns:
        Dict with processing results
    """
    if output_path is None:
        output_path = input_path

    results = {
        "cropped": False,
        "rotated": False,
        "rotation_degrees": 0,
        "ratio_info": None,
        "error": None,
    }

    try:
        # Step 1: Auto-crop to photo region
        results["cropped"] = auto_crop_photo(input_path, output_path)

        # When a distinct output path was requested but cropping found nothing,
        # copy the source over so later steps operate on the output and the
        # original is never modified.
        if output_path != input_path and not results["cropped"]:
            shutil.copy2(input_path, output_path)

        # Step 2: Auto-orient using AI (always on output_path — see above)
        if use_ai_orientation:
            rotation = auto_orient_photo(output_path)
            results["rotated"] = rotation != 0
            results["rotation_degrees"] = rotation

        # Step 3: Get ratio info for the final image
        results["ratio_info"] = get_photo_ratio_info(output_path)

    except Exception as e:
        results["error"] = str(e)

    return results


def detect_which_is_front(image_a_path: Path, image_b_path: Path) -> str:
    """
    Use AI to determine which of two images is the photo front vs back.

    In duplex scanning, one side has the actual photo, the other side
    might be blank, have handwriting, or stamps.

    Args:
        image_a_path: First image (scanner's "front")
        image_b_path: Second image (scanner's "back")

    Returns:
        "a" if image_a is the photo front, "b" if image_b is the photo front
    """
    try:
        import anthropic
    except ImportError:
        return "a"  # Default to first image

    api_key = get_anthropic_api_key()
    if not api_key:
        print("No Anthropic API key found for front/back detection")
        return "a"

    prompt = """I'm showing you two scans from opposite sides of the same photograph.

One side is the FRONT of the photo (the actual photograph with people, scenery, etc.)
The other side is the BACK of the photo (usually blank, or might have handwriting, dates, stamps, or processing marks).

Which image shows the FRONT of the photograph (the actual picture)?

Answer with ONLY the word "first" or "second" - nothing else."""

    try:
        # Load and resize both images
        def prepare_image(path):
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            max_dim = 512
            width, height = img.size
            if width > max_dim or height > max_dim:
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        image_a_base64 = prepare_image(image_a_path)
        image_b_base64 = prepare_image(image_b_path)

        from photopipe.config import get_config

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=get_config().vlm.model,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Image 1 (first):",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_a_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Image 2 (second):",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        result = response.content[0].text.strip().lower()

        if "first" in result:
            return "a"
        elif "second" in result:
            return "b"
        else:
            return "a"  # Default

    except Exception as e:
        print(f"Front/back detection error: {e}")
        return "a"


def batch_process_photos(input_folder: Path, pattern: str = "*.jpg") -> dict:
    """
    Process all photos in a folder.

    Args:
        input_folder: Folder containing scanned images
        pattern: Glob pattern for images

    Returns:
        Summary of processing results
    """
    summary = {
        "total": 0,
        "cropped": 0,
        "rotated": 0,
        "errors": 0,
    }

    for image_path in input_folder.glob(pattern):
        summary["total"] += 1
        results = process_scanned_photo(image_path)

        if results["cropped"]:
            summary["cropped"] += 1
        if results["rotated"]:
            summary["rotated"] += 1
        if results["error"]:
            summary["errors"] += 1

    return summary
