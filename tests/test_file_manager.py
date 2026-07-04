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
