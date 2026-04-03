from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.preprocess import (
    build_access_output_path,
    build_output_path,
    crop_borders,
    load_image,
    preprocess,
    process_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "raw" / "tamil_stone" / "IMG_3941.jpg"


class PreprocessTests(unittest.TestCase):
    def test_load_image_reads_sample(self) -> None:
        image = load_image(str(SAMPLE_IMAGE))
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)

    def test_crop_borders_removes_synthetic_frame(self) -> None:
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[20:80, 20:80] = (120, 120, 120)
        bordered = cv2.copyMakeBorder(image, 15, 15, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cropped = crop_borders(bordered, threshold=10)

        self.assertLess(cropped.shape[0], bordered.shape[0])
        self.assertLess(cropped.shape[1], bordered.shape[1])

    def test_preprocess_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "preprocessed.tif"
            result = preprocess(str(SAMPLE_IMAGE), str(output_path), save_access_copy=True)

            self.assertTrue(output_path.exists())
            self.assertTrue(build_access_output_path(output_path).exists())
            self.assertGreater(result.shape[0], 0)
            self.assertGreater(result.shape[1], 0)

            with Image.open(output_path) as image:
                self.assertGreater(image.size[0], 0)
                self.assertGreater(image.size[1], 0)

    def test_preprocess_no_access_copy(self) -> None:
        """Test that access copy can be disabled for efficiency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "preprocessed.tif"
            result = preprocess(str(SAMPLE_IMAGE), str(output_path), save_access_copy=False)

            self.assertTrue(output_path.exists())
            self.assertFalse(build_access_output_path(output_path).exists())
            self.assertGreater(result.shape[0], 0)

    def test_preprocess_deskew_disabled(self) -> None:
        """Test that deskew can be disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "preprocessed.tif"
            result = preprocess(
                str(SAMPLE_IMAGE), 
                str(output_path),
                enable_deskew=False,
                save_access_copy=False,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(result.shape[0], 0)

    def test_process_directory_writes_expected_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "input"
            output_dir = Path(tmp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            sample_copy = input_dir / SAMPLE_IMAGE.name
            shutil.copy2(SAMPLE_IMAGE, sample_copy)

            output_paths = process_directory(
                str(input_dir), 
                str(output_dir),
                save_access_copy=True,
            )

            self.assertEqual(len(output_paths), 1)
            expected_path = build_output_path(sample_copy, output_dir)
            self.assertEqual(output_paths[0], expected_path)
            self.assertTrue(expected_path.exists())
            self.assertTrue(build_access_output_path(expected_path).exists())


if __name__ == "__main__":
    unittest.main()
