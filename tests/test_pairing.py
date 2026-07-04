"""Tests for photopipe.pairing — front/back matching against real FastFoto names."""

from pathlib import Path

from photopipe.pairing import (
    extract_sequence_number,
    is_back_image,
    scan_input_folder,
)


class TestExtractSequenceNumber:
    def test_fastfoto_default_pattern(self):
        assert extract_sequence_number("FastFoto_0001", "{name}_{num}") == 1

    def test_custom_prefix(self):
        assert extract_sequence_number("photo_0042", "{name}_{num}") == 42

    def test_multi_underscore_name(self):
        assert extract_sequence_number("grandmas_album_p3_0007", "{name}_{num}") == 7

    def test_pattern_with_extension_is_stripped(self):
        assert extract_sequence_number("photo_0001", "{name}_{num}.jpg") == 1

    def test_back_image_does_not_match_front_pattern(self):
        assert extract_sequence_number("FastFoto_0001_b", "{name}_{num}") is None

    def test_no_match_returns_none(self):
        assert extract_sequence_number("IMG20240101", "{name}_{num}") is None

    def test_sequence_zero(self):
        assert extract_sequence_number("photo_0000", "{name}_{num}") == 0

    def test_case_insensitive(self):
        assert extract_sequence_number("PHOTO_0003", "{name}_{num}") == 3


class TestIsBackImage:
    def test_fastfoto_back(self):
        assert is_back_image("FastFoto_0001_b", "{name}_{num}_b") is True

    def test_front_is_not_back(self):
        assert is_back_image("FastFoto_0001", "{name}_{num}_b") is False

    def test_back_pattern_with_extension(self):
        assert is_back_image("photo_0001_b", "{name}_{num}_b.jpg") is True

    def test_uppercase_suffix(self):
        assert is_back_image("photo_0001_B", "{name}_{num}_b") is True


class TestScanInputFolder:
    def _make(self, folder: Path, *names: str) -> None:
        for name in names:
            (folder / name).write_bytes(b"\xff\xd8\xff\xe0test")

    def test_pairs_fronts_and_backs(self, tmp_path):
        self._make(
            tmp_path,
            "FastFoto_0001.jpg",
            "FastFoto_0001_b.jpg",
            "FastFoto_0002.jpg",
        )
        pairs = scan_input_folder(tmp_path, "{name}_{num}", "{name}_{num}_b")
        assert set(pairs.keys()) == {1, 2}
        assert pairs[1]["front"].name == "FastFoto_0001.jpg"
        assert pairs[1]["back"].name == "FastFoto_0001_b.jpg"
        assert pairs[2]["front"].name == "FastFoto_0002.jpg"
        assert pairs[2]["back"] is None

    def test_config_default_patterns_with_extension(self, tmp_path):
        self._make(tmp_path, "photo_0001.jpg", "photo_0001_b.jpg")
        pairs = scan_input_folder(tmp_path, "{name}_{num}.jpg", "{name}_{num}_b.jpg")
        assert 1 in pairs
        assert pairs[1]["front"] is not None
        assert pairs[1]["back"] is not None

    def test_skips_hidden_and_appledouble_files(self, tmp_path):
        self._make(tmp_path, "photo_0001.jpg", "._photo_0001.jpg", ".DS_Store")
        pairs = scan_input_folder(tmp_path, "{name}_{num}", "{name}_{num}_b")
        assert pairs[1]["front"].name == "photo_0001.jpg"
        assert len(pairs) == 1

    def test_non_image_files_ignored(self, tmp_path):
        self._make(tmp_path, "photo_0001.jpg")
        (tmp_path / "notes.txt").write_text("not an image")
        pairs = scan_input_folder(tmp_path, "{name}_{num}", "{name}_{num}_b")
        assert len(pairs) == 1
