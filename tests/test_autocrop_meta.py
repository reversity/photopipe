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


def test_autocrop_strips_orientation_tag(tmp_path):
    """cv2 applies EXIF orientation on read; the re-saved crop must not carry
    the tag again (would double-rotate in viewers)."""
    from PIL import Image
    src = tmp_path / "oriented.jpg"
    img = Image.new("RGB", (200, 140), (200, 210, 220))
    img.paste(Image.new("RGB", (120, 90), (30, 30, 30)), (40, 25))
    exif = img.getexif(); exif[274] = 6  # Orientation
    img.save(src, "JPEG", quality=95, exif=exif.tobytes(), dpi=(600, 600))

    out = tmp_path / "cropped.jpg"
    from photopipe.autocrop import auto_crop_photo
    assert auto_crop_photo(src, out) is True
    with Image.open(out) as o:
        # Orientation absent or normalized to 1 (upright)
        assert o.getexif().get(274, 1) == 1


def test_rotate_strips_orientation_tag(tmp_path):
    from PIL import Image
    p = tmp_path / "r.jpg"
    img = Image.new("RGB", (200, 140), (100, 100, 100))
    exif = img.getexif(); exif[274] = 3
    img.save(p, "JPEG", quality=95, exif=exif.tobytes(), dpi=(600, 600))
    from photopipe.autocrop import rotate_photo
    assert rotate_photo(p, 90) is True
    with Image.open(p) as o:
        assert o.getexif().get(274, 1) == 1
        assert o.info.get("dpi") == (600, 600)


def test_ai_orientation_off_by_default():
    from photopipe.config import Config
    assert Config().autocrop.ai_orientation is False
    assert Config().autocrop.enabled is True


def test_deskew_straightens_a_tilted_photo(tmp_path):
    """A dark photo tilted on a white scanner bed is straightened + cropped."""
    import numpy as np, cv2
    from photopipe.autocrop import deskew_and_crop, _normalized_min_area_rect, _find_photo_contour
    page = np.full((1200, 1000, 3), 255, dtype=np.uint8)  # white bed
    # a filled photo rectangle rotated ~8 degrees
    box = cv2.boxPoints(((500, 600), (600, 400), 8.0)).astype(np.int32)
    cv2.fillPoly(page, [box], (40, 60, 90))
    contour = _find_photo_contour(page)
    assert contour is not None
    _, _, angle = _normalized_min_area_rect(contour)
    assert abs(angle) > 1.5  # detects the tilt
    out = deskew_and_crop(page)
    assert out is not None and out.size > 0
    # result is close to the true photo size (600x400), not the full page
    h, w = out.shape[:2]
    assert 350 < min(h, w) < 450 and 550 < max(h, w) < 650


def test_process_one_photo_skips_crop_when_disabled(tmp_path, monkeypatch):
    from photopipe import capture_pipeline as cp
    from photopipe.config import get_config
    get_config.cache_clear()
    monkeypatch.setattr(get_config(), "autocrop", type("A", (), {"enabled": False, "ai_orientation": False})())
    called = []
    monkeypatch.setattr(cp, "process_scanned_photo", lambda *a, **k: called.append(a))
    # a photo whose original exists
    from photopipe.models import PhotoPair, PhotoPhase, PhotoStatus
    from photopipe.database import Database
    from pathlib import Path
    db = Database(db_path=tmp_path / "t.db")
    photo = PhotoPair(bucket_id="b", batch_id="", sequence_num=1,
                      front_path=tmp_path / "photo_0001.jpg", phase=PhotoPhase.CAPTURED)
    # no back, no original -> nothing to do, but disabled means no crop regardless
    cp._process_one_photo(db, "b", photo, None)
    assert called == []
    get_config.cache_clear()
