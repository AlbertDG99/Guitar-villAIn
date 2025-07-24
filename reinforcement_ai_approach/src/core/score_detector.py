#!/usr/bin/env python3
"""
Score Detector - Score Detection
================================

Module for detecting and reading the game score using OCR.
"""

import cv2
import numpy as np
import pytesseract

from src.utils.logger import setup_logger


class ScoreDetector:
    """
    Class for detecting and managing the game score.
    Maintains the current score state and updates it.
    """

    def __init__(self, score_region=None):
        """
        Initializes the score detector.

        Args:
            score_region (dict, optional): Screen region where the score is located.
        """
        self.logger = setup_logger("ScoreDetector")
        self.current_score = 0
        self.region = score_region
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789'

    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image to improve OCR accuracy.
        The score is usually white/bright text on a dark background.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh_image = cv2.threshold(
            gray_image, 180, 255, cv2.THRESH_BINARY_INV)

        return thresh_image

        return thresh_image

    def update_score(self, frame: np.ndarray) -> int:
        """
        Takes the complete game frame, crops the score region,
        detects the number and updates the internal score.
        """
        if self.region is None or frame is None or frame.size == 0:
            return self.current_score

        x, y, w, h = self.region['x'], self.region['y'], self.region['width'], self.region['height']
        image = frame[y:y + h, x:x + w]

        if image.size == 0:
            return self.current_score

        try:
            processed_image = self._preprocess_image_for_ocr(image)

            extracted_text = pytesseract.image_to_string(
                processed_image, config=self.tesseract_config)

            cleaned_text = "".join(filter(str.isdigit, extracted_text))

            if cleaned_text:
                new_score = int(cleaned_text)

                if new_score > self.current_score:
                    self.logger.info(
                        "Score updated: %d -> %d",
                        self.current_score,
                        new_score)
                    self.current_score = new_score

        except pytesseract.TesseractNotFoundError:
            self.logger.error(
                "Tesseract is not installed or not found in the system PATH.")
            self.logger.error(
                "Please install Tesseract-OCR and make sure it can be called from the terminal.")
            raise
        except Exception as e:
            self.logger.error("Error detecting score: %s", e)

        return self.current_score
