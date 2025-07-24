import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict


class ComboDetector:
    """
    Detects the combo multiplier from a screen region using OCR,
    with logic to avoid erroneous readings.
    """

    def __init__(self, config_region: Optional[Dict[str, int]]):
        """
        Initializes the combo detector.

        Args:
            config_region (Optional[Dict[str, int]]):
                Dictionary with the combo region: {'x', 'y', 'width', 'height'}.
                If None, the detector will be disabled.
        """
        if config_region:
            self.x = config_region['x']
            self.y = config_region['y']
            self.width = config_region['width']
            self.height = config_region['height']
            self.enabled = True
        else:
            self.enabled = False
            self.x = self.y = self.width = self.height = 0

        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        self.reset()

    def reset(self):
        """Resets the internal state of the detector."""
        self.last_valid_combo = 1

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image to improve OCR accuracy.
        """
        if image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale_factor = 3
        resized = cv2.resize(
            gray,
            (self.width *
             scale_factor,
             self.height *
             scale_factor),
            interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def detect(self, frame: np.ndarray) -> int:
        """
        Analyzes the frame, reads the combo text and returns the numerical multiplier.

        Returns:
            int: The combo multiplier (1, 2, 3, 4). Returns 1 if there's no combo or nothing is detected.
        """
        if not self.enabled or frame is None:
            return self.last_valid_combo

        roi = frame[self.y:self.y + self.height, self.x:self.x + self.width]
        if roi.size == 0:
            return self.last_valid_combo

        processed_roi = self._preprocess_for_ocr(roi)

        try:
            text = pytesseract.image_to_string(
                processed_roi, config=self.tesseract_config).strip()

            if not text:
                return self.last_valid_combo

            nums = "".join(filter(str.isdigit, text))

            if nums:
                new_combo = int(nums)

                if new_combo > self.last_valid_combo + 10:
                    return self.last_valid_combo

                self.last_valid_combo = new_combo
                return new_combo
            else:
                return self.last_valid_combo

        except Exception:
            return self.last_valid_combo

    def get_combo_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns the combo ROI for debugging."""
        if not self.enabled:
            return None
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]
