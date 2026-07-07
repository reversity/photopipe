"""Autocrop must preserve the scan DPI so crops don't report as 1 px/inch."""
import numpy as np
from PIL import Image

from photopipe.autocrop import auto_crop_photo, rotate_photo, _read_image_meta


def _scan_like(path, dpi=(600, 600)):
    """A page-like image: dark photo region on a light scanner bed, saved with DPI."""
    arr = np.full((900, 700, 3), 210, dtype=np.uint8)  # light "bed"
    arr[150:750, 120:580] = 40  # dark photo region
    Image.fromarray(arr).save(path, "JPEG", quality=95, dpi=dpi)


def test_read_image_meta_reads_dpi(tmp_path):
    p = tmp_path / "scan.jpg"
    _scan_like(p, dpi=(600, 600))
    _exif, dpi = _read_image_meta(p)
    assert dpi == (600, 600)


def test_read_image_meta_treats_1dpi_as_unknown(tmp_path):
    p = tmp_path / "flat.jpg"
    Image.new("RGB", (100, 100)).save(p, "JPEG")  # PIL default density 1,1
    _exif, dpi = _read_image_meta(p)
    assert dpi is None


def test_autocrop_preserves_dpi(tmp_path):
    src = tmp_path / "scan.jpg"
    out = tmp_path / "cropped.jpg"
    _scan_like(src, dpi=(600, 600))
    assert auto_crop_photo(src, out) is True
    with Image.open(out) as img:
        assert img.info.get("dpi") == (600, 600)
        assert img.size[0] > 200 and img.size[1] > 200  # real pixels, not a thumbnail


def test_rotate_preserves_dpi(tmp_path):
    p = tmp_path / "scan.jpg"
    _scan_like(p, dpi=(600, 600))
    assert rotate_photo(p, 90) is True
    with Image.open(p) as img:
        assert img.info.get("dpi") == (600, 600)
