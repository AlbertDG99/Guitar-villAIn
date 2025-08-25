#!/usr/bin/env python3
"""
Score Detector - Score Detection
================================

Module for detecting and reading the game score using OCR.
"""

import cv2
import numpy as np
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

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

        # Load optimized parameters if available
        self.optimized_params = self._load_optimized_params()

        # Initialize EasyOCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for better compatibility
                self.logger.info("EasyOCR initialized for better text recognition")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None

    def _load_optimized_params(self):
        """Load optimized OCR parameters from config file"""
        import configparser
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "config" / "config.ini"
        config = configparser.ConfigParser()

        try:
            config.read(config_path)
            if config.has_section('OCR_OPTIMIZATION'):
                params = {}
                for key in ['brightness', 'contrast', 'sharpness', 'gaussian_blur',
                           'adaptive_block_size', 'adaptive_c', 'psm_mode', 'oem_mode',
                           'easyocr_confidence', 'min_digits', 'max_digits',
                           'max_score_change', 'min_score_change']:
                    if key in config['OCR_OPTIMIZATION']:
                        value = config['OCR_OPTIMIZATION'][key]
                        # Convert to appropriate type
                        if key in ['brightness', 'gaussian_blur', 'adaptive_block_size',
                                 'adaptive_c', 'psm_mode', 'oem_mode', 'min_digits', 'max_digits']:
                            params[key] = int(value)
                        elif key in ['contrast', 'sharpness', 'easyocr_confidence']:
                            params[key] = float(value)
                        elif key in ['max_score_change', 'min_score_change']:
                            params[key] = int(value)

                self.logger.info("Loaded optimized OCR parameters from config")
                return params
            else:
                self.logger.info("No optimized OCR parameters found in config, using defaults")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load optimized OCR parameters: {e}")
            return {}

    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image to improve OCR accuracy without distorting text.
        Uses optimized parameters if available, otherwise defaults.
        """
        # Get parameters (optimized or default)
        brightness = self.optimized_params.get('brightness', 10)
        contrast = self.optimized_params.get('contrast', 1.2)
        sharpness = self.optimized_params.get('sharpness', 1.0)
        gaussian_blur = self.optimized_params.get('gaussian_blur', 3)
        adaptive_block_size = self.optimized_params.get('adaptive_block_size', 11)
        adaptive_c = self.optimized_params.get('adaptive_c', 2)

        # Start with the original image to preserve text characteristics
        processed = image.copy()

        # Step 1: Enhance image quality (brightness, contrast, sharpness)
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Enhance lightness channel with optimized parameters
        l = cv2.add(l, brightness)
        l = cv2.multiply(l, contrast)
        l = np.clip(l, 0, 255).astype(np.uint8)

        # Merge back and convert to BGR
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Step 2: Apply sharpening with optimized strength
        if sharpness > 0:
            kernel = np.array([[-1,-1,-1],
                              [-1, 1+8*sharpness,-1],
                              [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)

        # Step 3: Convert to grayscale for OCR
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Step 4: Apply Gaussian blur with optimized kernel size
        ksize = max(1, gaussian_blur if gaussian_blur % 2 == 1 else gaussian_blur + 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Step 5: Use adaptive thresholding with optimized parameters
        block_size = max(3, adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, adaptive_c
        )

        # Step 6: Very mild morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh

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

            # Debug: Show what we're processing
            self.logger.debug("Processing score region of size: %dx%d", image.shape[1], image.shape[0])
            self.logger.debug("Score region coords: x=%d, y=%d, w=%d, h=%d",
                            self.region['x'], self.region['y'], self.region['width'], self.region['height'])

            # Use optimized parameters or defaults
            psm_mode = self.optimized_params.get('psm_mode', 7)
            oem_mode = self.optimized_params.get('oem_mode', 3)

            # OCR configurations with optimized parameters
            configs = [
                f'--psm {psm_mode} --oem {oem_mode} -c tessedit_char_whitelist=0123456789',
                '--psm 7 -c tessedit_char_whitelist=0123456789',  # Fallback simple mode
                '--psm 8 -c tessedit_char_whitelist=0123456789',  # Fallback word mode
            ]

            try:
                # Try Tesseract with different configurations
                best_cleaned = ""
                best_text = ""

                for i, config in enumerate(configs):
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config)
                        cleaned = ''.join(filter(str.isdigit, text))

                        if len(cleaned) > len(best_cleaned):
                            best_cleaned = cleaned
                            best_text = text

                        # Debug log for first config only
                        if i == 0:
                            if cleaned:
                                self.logger.debug("Tesseract raw text: '%s', cleaned: '%s'", text.strip(), cleaned)
                            else:
                                self.logger.debug("Tesseract raw text: '%s', no digits found", text.strip())
                    except Exception as e:
                        self.logger.debug("Tesseract config %d failed: %s", i, e)
                        continue

                # Try EasyOCR as backup if available and Tesseract didn't find enough digits
                easyocr_confidence = self.optimized_params.get('easyocr_confidence', 0.5)
                min_digits = self.optimized_params.get('min_digits', 1)

                if self.easyocr_reader and len(best_cleaned) < min_digits:
                    try:
                        # EasyOCR works better with the original image (before heavy processing)
                        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        results = self.easyocr_reader.readtext(original_gray)

                        for (bbox, text, confidence) in results:
                            if confidence > easyocr_confidence:  # Use optimized confidence threshold
                                cleaned = ''.join(filter(str.isdigit, text))
                                if len(cleaned) > len(best_cleaned):
                                    best_cleaned = cleaned
                                    best_text = text
                                    self.logger.debug("EasyOCR found: '%s' with confidence %.2f", text, confidence)
                    except Exception as e:
                        self.logger.debug("EasyOCR failed: %s", e)

                cleaned_text = best_cleaned
                text = best_text

                # Only update if we got a valid number with reasonable length
                max_digits = self.optimized_params.get('max_digits', 6)
                if min_digits <= len(cleaned_text) <= max_digits:
                    try:
                        new_score = int(cleaned_text)
                        # Use optimized score change limits
                        max_score_change = self.optimized_params.get('max_score_change', 20000)
                        min_score_change = self.optimized_params.get('min_score_change', -100)

                        score_change = new_score - self.current_score

                        if (score_change > 0 and score_change <= max_score_change) or (score_change < 0 and score_change >= min_score_change):
                            self.logger.info(
                                "Score updated: %d -> %d (OCR: '%s')",
                                self.current_score,
                                new_score,
                                cleaned_text)
                            self.current_score = new_score
                        else:
                            self.logger.debug("Score change too extreme, ignoring: %d -> %d (change: %d)",
                                            self.current_score, new_score, score_change)
                    except ValueError:
                        self.logger.warning("Could not convert OCR text to number: '%s'", cleaned_text)
                else:
                    if len(cleaned_text) > 6:
                        self.logger.debug("OCR text too long (%d chars), ignoring: '%s'", len(cleaned_text), cleaned_text)
                    # Don't log when no digits found - too noisy

            except Exception as e:
                self.logger.debug("OCR processing failed: %s", e)

        except pytesseract.TesseractNotFoundError:
            self.logger.error(
                "Tesseract is not installed or not found in the system PATH.")
            self.logger.error(
                "Please install Tesseract-OCR and make sure it can be called from the terminal.")
            raise
        except Exception as e:
            self.logger.error("Error detecting score: %s", e)

        return self.current_score

    def get_processed_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the processed image for debugging purposes.
        """
        if self.region is None or frame is None or frame.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        x, y, w, h = self.region['x'], self.region['y'], self.region['width'], self.region['height']
        image = frame[y:y + h, x:x + w]

        if image.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        return self._preprocess_image_for_ocr(image)
