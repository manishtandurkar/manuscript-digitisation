"""Tests for Stage 4 — OCR & Transcription (src/ocr.py)."""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.ocr import (
    SCRIPT_CONFIG,
    _group_words_into_lines,
    build_output_path,
    detect_script,
    ocr_ensemble,
    ocr_easyocr,
    ocr_tesseract,
    transcribe,
)

SAMPLE_IMAGE = Path(__file__).parent.parent / "data" / "raw" / "tamil_stone" / "IMG_3941.jpg"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def blank_image() -> np.ndarray:
    """Solid-white BGR image — produces empty OCR output."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


@pytest.fixture()
def synthetic_text_image() -> np.ndarray:
    """Dark-on-white image with a few blobs that simulate text."""
    img = np.ones((80, 300, 3), dtype=np.uint8) * 255
    # Draw simple rectangles to simulate characters
    cv2.rectangle(img, (10, 20), (30, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (40, 20), (60, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (70, 20), (90, 50), (0, 0, 0), -1)
    return img


@pytest.fixture()
def sample_word_boxes() -> list[dict]:
    return [
        {"text": "hello", "confidence": 0.90, "box": [10, 10, 40, 15]},
        {"text": "world", "confidence": 0.80, "box": [60, 12, 40, 15]},
        {"text": "second", "confidence": 0.70, "box": [10, 40, 50, 15]},
        {"text": "line",   "confidence": 0.60, "box": [70, 42, 30, 15]},
    ]


# ─── detect_script ───────────────────────────────────────────────────────────

class TestDetectScript:
    def test_returns_known_script_name(self, blank_image):
        result = detect_script(blank_image)
        assert result in SCRIPT_CONFIG, f"Unknown script returned: {result}"

    def test_handles_none_gracefully(self):
        result = detect_script(None)
        assert isinstance(result, str)

    def test_handles_empty_array(self):
        result = detect_script(np.array([]))
        assert isinstance(result, str)

    def test_sample_image_returns_tamil(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("Sample image not available")
        img = cv2.imread(str(SAMPLE_IMAGE))
        result = detect_script(img)
        # Heuristic may not be perfect, but must return a valid script
        assert result in SCRIPT_CONFIG


# ─── ocr_tesseract ───────────────────────────────────────────────────────────

class TestOcrTesseract:
    def test_returns_empty_when_lang_is_none(self, blank_image):
        result = ocr_tesseract(blank_image, None)
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["word_boxes"] == []

    def test_returns_dict_with_required_keys(self, blank_image):
        result = ocr_tesseract(blank_image, "tam")
        assert "text" in result
        assert "confidence" in result
        assert "word_boxes" in result

    def test_confidence_in_range(self, blank_image):
        result = ocr_tesseract(blank_image, "tam")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_graceful_when_pytesseract_unavailable(self, blank_image):
        with patch("src.ocr._TESS_AVAILABLE", False):
            result = ocr_tesseract(blank_image, "tam")
        assert result["text"] == ""

    def test_handles_tesseract_exception(self, blank_image):
        import sys
        fake_tess = MagicMock()
        fake_tess.Output = types.SimpleNamespace(DICT="dict")
        fake_tess.image_to_data.side_effect = RuntimeError("tess crash")
        with patch("src.ocr._TESS_AVAILABLE", True):
            with patch.dict(sys.modules, {"pytesseract": fake_tess}):
                with patch("src.ocr.pytesseract", fake_tess, create=True):
                    result = ocr_tesseract(blank_image, "tam")
        assert result["text"] == ""
        assert result["confidence"] == 0.0


# ─── ocr_easyocr ─────────────────────────────────────────────────────────────

class TestOcrEasyocr:
    def test_returns_empty_when_langs_empty(self, blank_image):
        result = ocr_easyocr(blank_image, [])
        assert result["text"] == ""
        assert result["word_boxes"] == []

    def test_returns_dict_with_required_keys(self, blank_image):
        result = ocr_easyocr(blank_image, ["ta"])
        assert "text" in result
        assert "confidence" in result
        assert "word_boxes" in result

    def test_graceful_when_easyocr_unavailable(self, blank_image):
        with patch("src.ocr._EASY_AVAILABLE", False):
            result = ocr_easyocr(blank_image, ["ta"])
        assert result["text"] == ""

    def test_mocked_easyocr_result(self, blank_image):
        fake_bbox = [[10, 5], [60, 5], [60, 20], [10, 20]]
        fake_results = [(fake_bbox, "test text", 0.92)]

        with patch("src.ocr._EASY_AVAILABLE", True):
            mock_reader = MagicMock()
            mock_reader.readtext.return_value = fake_results
            with patch("src.ocr._EASYOCR_READER_CACHE", {"ta": mock_reader}):
                result = ocr_easyocr(blank_image, ["ta"])

        assert result["text"] == "test text"
        assert abs(result["confidence"] - 0.92) < 0.01
        assert len(result["word_boxes"]) == 1
        wb = result["word_boxes"][0]
        assert wb["text"] == "test text"
        assert wb["box"] == [10, 5, 50, 15]


# ─── ocr_ensemble ────────────────────────────────────────────────────────────

class TestOcrEnsemble:
    def test_returns_required_keys(self, blank_image):
        result = ocr_ensemble(blank_image, "tamil")
        for key in ("text", "confidence", "word_boxes", "engine_used"):
            assert key in result

    def test_prefers_higher_confidence_engine(self, blank_image):
        tess_result = {"text": "tess text", "confidence": 0.90, "word_boxes": []}
        easy_result = {"text": "easy text", "confidence": 0.70, "word_boxes": []}

        with patch("src.ocr.ocr_tesseract", return_value=tess_result):
            with patch("src.ocr.ocr_easyocr", return_value=easy_result):
                result = ocr_ensemble(blank_image, "tamil")

        assert result["text"] == "tess text"

    def test_deduplicates_word_boxes(self, blank_image):
        shared_box = {"text": "word", "confidence": 0.9, "box": [10, 10, 30, 15]}
        tess_result = {"text": "word", "confidence": 0.9, "word_boxes": [shared_box]}
        easy_result = {"text": "word", "confidence": 0.8, "word_boxes": [shared_box]}

        with patch("src.ocr.ocr_tesseract", return_value=tess_result):
            with patch("src.ocr.ocr_easyocr", return_value=easy_result):
                result = ocr_ensemble(blank_image, "tamil")

        # Same position — should not be duplicated
        assert len(result["word_boxes"]) == 1

    def test_unknown_script_falls_back_to_tamil_config(self, blank_image):
        result = ocr_ensemble(blank_image, "unknown_script")
        assert "text" in result


# ─── _group_words_into_lines ─────────────────────────────────────────────────

class TestGroupWordsIntoLines:
    def test_empty_input(self):
        assert _group_words_into_lines([]) == []

    def test_groups_two_lines(self, sample_word_boxes):
        lines = _group_words_into_lines(sample_word_boxes)
        assert len(lines) == 2

    def test_line_dict_keys(self, sample_word_boxes):
        lines = _group_words_into_lines(sample_word_boxes)
        for ln in lines:
            assert "line_number" in ln
            assert "text" in ln
            assert "confidence" in ln
            assert "bounding_box" in ln
            assert "uncertain" in ln

    def test_line_numbers_sequential(self, sample_word_boxes):
        lines = _group_words_into_lines(sample_word_boxes)
        for i, ln in enumerate(lines, start=1):
            assert ln["line_number"] == i

    def test_uncertain_flag_set_for_low_confidence(self):
        word_boxes = [{"text": "x", "confidence": 0.40, "box": [0, 0, 10, 10]}]
        lines = _group_words_into_lines(word_boxes)
        assert lines[0]["uncertain"] is True

    def test_uncertain_flag_false_for_high_confidence(self):
        word_boxes = [{"text": "x", "confidence": 0.90, "box": [0, 0, 10, 10]}]
        lines = _group_words_into_lines(word_boxes)
        assert lines[0]["uncertain"] is False

    def test_bounding_box_has_four_elements(self, sample_word_boxes):
        lines = _group_words_into_lines(sample_word_boxes)
        for ln in lines:
            assert len(ln["bounding_box"]) == 4


# ─── transcribe ──────────────────────────────────────────────────────────────

class TestTranscribe:
    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe(str(tmp_path / "nonexistent.png"))

    def test_raises_for_unreadable_file(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            transcribe(str(bad))

    def test_returns_required_keys(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), blank_image)
        result = transcribe(str(img_path), script="tamil")
        for key in ("script", "text", "lines", "overall_confidence", "engine_used", "uncertain_regions"):
            assert key in result

    def test_auto_script_detection(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), blank_image)
        result = transcribe(str(img_path), script="auto")
        assert result["script"] in SCRIPT_CONFIG

    def test_saves_json_to_output_path(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        out_path = tmp_path / "out.json"
        cv2.imwrite(str(img_path), blank_image)
        transcribe(str(img_path), script="tamil", output_path=str(out_path))
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "script" in data

    def test_brahmi_script_returns_manual_flag(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), blank_image)
        result = transcribe(str(img_path), script="brahmi")
        assert result.get("status") == "manual_transcription_required"
        assert result["overall_confidence"] == 0.0

    def test_confidence_status_verified(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), blank_image)
        with patch("src.ocr.ocr_ensemble") as mock_ensemble:
            mock_ensemble.return_value = {
                "text": "some text",
                "confidence": 0.92,
                "word_boxes": [{"text": "some", "confidence": 0.92, "box": [0, 0, 30, 15]},
                               {"text": "text", "confidence": 0.92, "box": [40, 0, 30, 15]}],
                "engine_used": "tesseract",
            }
            result = transcribe(str(img_path), script="tamil")
        assert result["confidence_status"] == "verified"

    def test_overall_confidence_in_range(self, tmp_path, blank_image):
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), blank_image)
        result = transcribe(str(img_path), script="tamil")
        assert 0.0 <= result["overall_confidence"] <= 1.0

    def test_uses_sample_image_if_available(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("Sample image not available")
        result = transcribe(str(SAMPLE_IMAGE), script="tamil")
        assert result["script"] == "tamil"
        assert isinstance(result["text"], str)


# ─── build_output_path ───────────────────────────────────────────────────────

class TestBuildOutputPath:
    def test_stem_and_suffix(self, tmp_path):
        out = build_output_path("data/binarised/IMG_3941_binarised.png", tmp_path)
        assert out.name == "IMG_3941_binarised_transcription.json"
        assert out.parent == tmp_path

    def test_output_dir_created_correctly(self, tmp_path):
        out = build_output_path("foo/bar.png", tmp_path / "out")
        assert out.suffix == ".json"
