"""Tests for photopipe.file_manager collision safety."""

from PIL import Image

from photopipe.file_manager import _archive_copy, generate_output_filename
from photopipe.models import Batch, PhotoPair


def _photo(tmp_path, batch, seq=1):
    front = tmp_path / f"photo_{seq:04d}.jpg"
    Image.new("RGB", (100, 80)).save(front)
    back = tmp_path / f"photo_{seq:04d}_b.jpg"
    Image.new("RGB", (100, 80), color=(200, 0, 0)).save(back)
    return PhotoPair(batch_id=batch.id, sequence_num=seq, front_path=front, back_path=back)


class TestGenerateOutputFilename:
    def test_template_without_side_gets_back_suffix(self, tmp_path, monkeypatch):
        from photopipe import file_manager

        config = file_manager.get_config()
        monkeypatch.setattr(
            config.output, "filename_template", "{date}_{batch_name}_{sequence:04d}"
        )
        batch = Batch(name="Trip 1985")
        photo = _photo(tmp_path, batch)
        front_name = generate_output_filename(photo, batch, "front")
        back_name = generate_output_filename(photo, batch, "back")
        assert front_name != back_name
        assert "_back" in back_name


class TestArchiveCopy:
    def test_identical_file_not_duplicated(self, tmp_path):
        src = tmp_path / "a.jpg"
        src.write_bytes(b"same-bytes")
        archive = tmp_path / "archive"
        archive.mkdir()
        first = _archive_copy(src, archive)
        second = _archive_copy(src, archive)
        assert first == second
        assert len(list(archive.iterdir())) == 1

    def test_different_content_same_name_gets_suffix(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "photo_0001.jpg").write_bytes(b"batch A pixels")

        src = tmp_path / "photo_0001.jpg"
        src.write_bytes(b"batch B pixels - different")
        dest = _archive_copy(src, archive)

        assert dest.name == "photo_0001_1.jpg"
        assert dest.read_bytes() == b"batch B pixels - different"
        # Batch A's archived copy is untouched
        assert (archive / "photo_0001.jpg").read_bytes() == b"batch A pixels"


class TestFinalizeReviewGate:
    def test_auto_approve_does_not_finalize_low_confidence_review_photos(self, tmp_path, monkeypatch):
        """auto_approve_high_confidence must only finalize the photos it approved,
        not every needs-review photo."""
        from photopipe.database import Database
        from photopipe.file_manager import finalize_batch
        from photopipe.models import Batch, PhotoPair, PhotoStatus, DateConfidence
        from datetime import date
        import photopipe.file_manager as fm

        db = Database(db_path=tmp_path / "t.db")
        batch = Batch(name="Mixed", date_start=date(1985, 1, 1), date_end=date(1985, 12, 31))
        db.create_batch(batch)

        # low-confidence photo still needing review
        low = PhotoPair(batch_id=batch.id, sequence_num=1,
                        front_path=tmp_path / "a.jpg", needs_review=True,
                        date_confidence=DateConfidence.LOW)
        from PIL import Image
        Image.new("RGB", (50, 40)).save(low.front_path)
        db.create_photo(low)

        # Don't actually write files/metadata
        monkeypatch.setattr(fm, "finalize_photo", lambda p, b, d, **k: p)

        report = finalize_batch(batch, db, auto_approve_high_confidence=True)
        # the low-confidence review photo is skipped, nothing finalized
        assert report.photo_count == 0


def test_update_photo_persists_path_change(tmp_path):
    """update_photo must save front_path/back_path (regression: it silently
    dropped them, so a file move left the DB pointing at the old location)."""
    from photopipe.database import Database
    from photopipe.models import PhotoPair
    db = Database(db_path=tmp_path / "t.db")
    photo = PhotoPair(batch_id="b", sequence_num=1,
                      front_path=tmp_path / "old" / "f.jpg",
                      back_path=tmp_path / "old" / "f_b.jpg")
    db.create_photo(photo)
    photo.front_path = tmp_path / "new" / "f.jpg"
    photo.back_path = tmp_path / "new" / "f_b.jpg"
    db.update_photo(photo)
    reloaded = db.get_photo(photo.id)
    assert str(reloaded.front_path).endswith("new/f.jpg")
    assert str(reloaded.back_path).endswith("new/f_b.jpg")
