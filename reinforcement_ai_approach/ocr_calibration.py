"""
OCR Calibration Tool - Professional OCR optimization for score detection
Optimized for GPU acceleration and maximum performance
"""

import cv2
import numpy as np
import time
from pathlib import Path
import configparser
import logging

# Professional imports with proper error handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("WARNING: EasyOCR not available. Install with: pip install easyocr")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Project imports with proper fallback
try:
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import setup_logger
    from src.core.screen_capture import ScreenCapture
    from src.core.score_detector import ScoreDetector
except ImportError:
    # Fallback imports
    import sys
    sys.path.append(str(Path(__file__).parent / "src"))
    try:
        from src.utils.config_manager import ConfigManager
        from src.utils.logger import setup_logger
        from src.core.screen_capture import ScreenCapture
        from src.core.score_detector import ScoreDetector
    except ImportError as e:
        print(f"ERROR: Could not import required modules: {e}")
        print("Please ensure you're running from the correct directory")
        sys.exit(1)


class OcrCalibrator:
    """
    Professional OCR Calibrator with GPU acceleration and performance optimization
    """

    def __init__(self):
        """Initialize the OCR calibrator with optimized settings"""
        self.logger = setup_logger("OcrCalibrator")
        self.logger.info("Initializing OCR Calibrator with GPU optimization...")

        # Initialize configuration
        config_path = Path(__file__).parent / "config" / "config.ini"
        self.cfg = ConfigManager(config_path=str(config_path))

        # Optimized parameters for maximum performance
        self.params = self._get_optimized_params()

        # Initialize OCR engine with GPU acceleration
        self.reader = self._initialize_ocr_engine()

        # Initialize screen capture
        self.cap = ScreenCapture(self.cfg)
        self.cap.start()

        # Initialize score detector
        score_region = self.cfg.get_score_region()
        self.score_detector = ScoreDetector(score_region)

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.ocr_times = []
        self.running = False

        # GPU status
        self.gpu_available = self._check_gpu_availability()

        self.logger.info(f"OCR Calibrator initialized. GPU: {self.gpu_available}")
        self._print_system_info()

    def _get_optimized_params(self):
        """Get optimized parameters for maximum performance"""
        return {
            # Image preprocessing - Optimized for speed and accuracy
            'brightness': 5,
            'contrast': 1.3,
            'sharpness': 0.5,
            'gaussian_blur': 1,

            # EasyOCR optimization parameters
            'min_confidence': 0.3,

            # Validation
            'min_digits': 1,
            'max_digits': 8,
            'max_score_change': 50000,
            'min_score_change': -1000,
        }

    def _check_gpu_availability(self):
        """Check if GPU is available and properly configured"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - GPU acceleration disabled")
            return False

        try:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU detected: {gpu_name}")
                return True
            else:
                self.logger.info("No GPU detected - using CPU mode")
                return False
        except Exception as e:
            self.logger.warning(f"Error checking GPU availability: {e}")
            return False

    def _initialize_ocr_engine(self):
        """Initialize EasyOCR with optimized settings for maximum performance"""
        if not EASYOCR_AVAILABLE:
            self.logger.error("EasyOCR not available. Please install with: pip install easyocr")
            return None

        try:
            # EasyOCR configuration - simplified for maximum compatibility
            gpu_enabled = self._check_gpu_availability()

            # Basic parameters that EasyOCR constructor accepts
            reader_kwargs = {
                'lang_list': ['en'],
                'gpu': gpu_enabled,
                'model_storage_directory': None,
                'user_network_directory': None,
                'recognizer': True,
                'detector': True,
                'verbose': True,  # Enable verbose to see GPU usage
            }

            self.logger.info(f"Initializing EasyOCR with GPU: {gpu_enabled}")
            self.logger.info("EasyOCR will use optimized settings for performance")

            reader = easyocr.Reader(**reader_kwargs)

            # Warm up the engine
            self._warm_up_engine(reader)

            self.logger.info("✅ EasyOCR initialized successfully with GPU acceleration")
            return reader

        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            return None

    def _warm_up_engine(self, reader):
        """Warm up the OCR engine for better performance"""
        try:
            # Create a small dummy image for warm-up
            dummy_img = np.full((100, 200, 3), 128, dtype=np.uint8)
            cv2.putText(dummy_img, "0123456789", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Run warm-up inference
            results = reader.readtext(dummy_img)
            self.logger.info("OCR engine warm-up completed")

        except Exception as e:
            self.logger.warning(f"OCR warm-up failed: {e}")

    def _print_system_info(self):
        """Print system information for debugging"""
        print(f"\n{'='*60}")
        print("OCR CALIBRATION TOOL - PROFESSIONAL VERSION")
        print(f"{'='*60}")
        print(f"GPU Available: {self.gpu_available}")
        print(f"EasyOCR Available: {EASYOCR_AVAILABLE}")
        print(f"PyTorch Available: {TORCH_AVAILABLE}")

        if TORCH_AVAILABLE:
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        print(f"GPU Acceleration: {'✅ ENABLED' if self.gpu_available else '❌ DISABLED'}")
        print(f"Performance Mode: OPTIMIZED")
        print(f"{'='*60}\n")

    def preprocess_image(self, image):
        """Optimized image preprocessing for maximum OCR accuracy"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply brightness and contrast
            if self.params['brightness'] != 0 or self.params['contrast'] != 1.0:
                gray = cv2.add(gray, self.params['brightness'])
                gray = cv2.multiply(gray, self.params['contrast'])
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            # Apply Gaussian blur if needed
            if self.params['gaussian_blur'] > 1:
                ksize = max(1, self.params['gaussian_blur'] if self.params['gaussian_blur'] % 2 == 1
                           else self.params['gaussian_blur'] + 1)
                gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

            # Apply sharpening if needed
            if self.params['sharpness'] > 0:
                kernel = np.array([[-1, -1, -1], [-1, 1+8*self.params['sharpness'], -1], [-1, -1, -1]])
                gray = cv2.filter2D(gray, -1, kernel)
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            return gray
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            return image

    def perform_ocr(self, frame):
        """Perform optimized OCR on the score region"""
        if not self.reader:
            return "OCR not initialized", 0.0

        start_time = time.time()

        try:
            # Get score region
            score_region = self.cfg.get_score_region()
            if not score_region:
                return "No score region configured", 0.0

            # Extract score region
            score_image = self._extract_score_region(frame, score_region)
            if score_image is None:
                return "Invalid score region", 0.0

            # Preprocess image
            processed_image = self.preprocess_image(score_image)

            # Perform OCR with optimized settings
            results = self.reader.readtext(processed_image)

            # Find best result
            best_text = ""
            best_confidence = 0.0

            for (bbox, text, confidence) in results:
                if confidence > self.params['min_confidence']:
                    # Extract only digits
                    cleaned = ''.join(filter(str.isdigit, text))
                    if len(cleaned) >= self.params['min_digits'] and confidence > best_confidence:
                        best_text = cleaned
                        best_confidence = confidence

            # Track performance
            ocr_time = time.time() - start_time
            self.ocr_times.append(ocr_time)
            if len(self.ocr_times) > 100:
                self.ocr_times.pop(0)

            return best_text, best_confidence

        except Exception as e:
            self.logger.error(f"OCR processing error: {e}")
            return "", 0.0

    def _extract_score_region(self, frame, score_region):
        """Extract the score region from the frame"""
        try:
            x, y = int(score_region['x']), int(score_region['y'])
            w, h = int(score_region['width']), int(score_region['height'])

            # Handle relative coordinates
            if (x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]):
                return frame[y:y+h, x:x+w]
            else:
                # Try absolute to relative conversion
                capture_config = self.cfg.get_capture_area_config()
                if capture_config:
                    left, top = int(capture_config['left']), int(capture_config['top'])
                    rel_x = x - left
                    rel_y = y - top

                    if (rel_x >= 0 and rel_y >= 0 and
                        rel_x + w <= frame.shape[1] and
                        rel_y + h <= frame.shape[0]):
                        return frame[rel_y:rel_y+h, rel_x:rel_x+w]

            return None
        except Exception as e:
            self.logger.error(f"Error extracting score region: {e}")
            return None

    def get_frame(self):
        """Get frame from screen capture"""
        return self.cap.get_latest_frame()

    def create_calibration_window(self):
        """Create a simple and efficient calibration window"""
        cv2.namedWindow('OCR Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('OCR Calibration', 1400, 800)  # Increased width for larger previews

        # Create essential trackbars only
        cv2.createTrackbar('Brightness', 'OCR Calibration', self.params['brightness'], 50, lambda x: None)
        cv2.createTrackbar('Contrast', 'OCR Calibration', int(self.params['contrast']*10), 20, lambda x: None)
        cv2.createTrackbar('Sharpness', 'OCR Calibration', int(self.params['sharpness']*10), 20, lambda x: None)
        cv2.createTrackbar('Blur', 'OCR Calibration', self.params['gaussian_blur'], 9, lambda x: None)
        cv2.createTrackbar('Min Conf', 'OCR Calibration', int(self.params['min_confidence']*100), 100, lambda x: None)

    def update_params(self):
        """Update parameters from trackbar positions"""
        try:
            self.params['brightness'] = cv2.getTrackbarPos('Brightness', 'OCR Calibration')
            self.params['contrast'] = cv2.getTrackbarPos('Contrast', 'OCR Calibration') / 10.0
            self.params['sharpness'] = cv2.getTrackbarPos('Sharpness', 'OCR Calibration') / 10.0
            self.params['gaussian_blur'] = max(1, cv2.getTrackbarPos('Blur', 'OCR Calibration'))
            self.params['min_confidence'] = cv2.getTrackbarPos('Min Conf', 'OCR Calibration') / 100.0
        except Exception as e:
            self.logger.debug(f"Parameter update error: {e}")

    def create_display_frame(self, frame, ocr_result, confidence):
        """Create the display frame with all information"""
        if frame is None:
            # Create error frame
            display = np.zeros((800, 1400, 3), dtype=np.uint8)  # Increased width for larger previews
            cv2.putText(display, "NO SIGNAL", (400, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            return display

        try:
            # Create display frame - LARGER for better visibility
            display = np.zeros((800, 1400, 3), dtype=np.uint8)  # Increased from 1200 to 1400 width

            # Left side: Game screen with score region highlighted
            game_area = self._create_game_preview(frame, display)

            # Right side: Processed image (what OCR sees)
            processed_area = self._create_processed_preview(frame, display)
            
            # Add OCR debug info
            self._add_ocr_debug_info(display, frame)

            # Add OCR results and info
            self._add_ocr_info(display, ocr_result, confidence)

            # Add parameter info
            self._add_parameter_info(display)

            # Add performance info
            self._add_performance_info(display)

            return display

        except Exception as e:
            self.logger.error(f"Display creation error: {e}")
            return np.zeros((800, 1200, 3), dtype=np.uint8)

    def _create_game_preview(self, frame, display):
        """Create game screen preview with score region highlighted"""
        try:
            # Calculate scaling to fit in left side - MAKE IT LARGER
            target_h, target_w = 600, 800  # Increased from 400x600 to 600x800
            frame_h, frame_w = frame.shape[:2]

            scale = min(target_w / frame_w, target_h / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)

            # Resize frame
            game_preview = cv2.resize(frame, (new_w, new_h))

            # Create black background
            bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Center the game preview
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = game_preview

            # Draw score region if available
            score_region = self.cfg.get_score_region()
            if score_region:
                x, y = int(score_region['x']), int(score_region['y'])
                w, h = int(score_region['width']), int(score_region['height'])

                # Scale coordinates
                scaled_x = int(x * scale) + x_offset
                scaled_y = int(y * scale) + y_offset
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)

                # Draw score region with thicker lines and better visibility
                cv2.rectangle(bg, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h),
                            (0, 255, 0), 3)  # Thicker line
                cv2.putText(bg, "SCORE REGION", (scaled_x, scaled_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Add coordinates info for debugging
                coord_text = f"({x},{y}) {w}x{h}"
                cv2.putText(bg, coord_text, (scaled_x, scaled_y + scaled_h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Copy to display - adjust position for larger preview
            display[0:600, 0:800] = bg
            return bg

        except Exception as e:
            self.logger.error(f"Game preview creation error: {e}")
            return None

    def _create_processed_preview(self, frame, display):
        """Create processed image preview - MAKE IT MUCH LARGER"""
        try:
            score_region = self.cfg.get_score_region()
            if not score_region:
                return

            # Extract and process score region
            score_image = self._extract_score_region(frame, score_region)
            if score_image is None:
                return
            
            processed = self.preprocess_image(score_image)

            # Convert to BGR for display
            if len(processed.shape) == 2:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed

            # Resize to fit - MUCH LARGER for better visibility
            target_h, target_w = 400, 600  # Increased from 150x300 to 400x600
            proc_h, proc_w = processed_bgr.shape[:2]

            scale = min(target_w / proc_w, target_h / proc_h)
            new_w = int(proc_w * scale)
            new_h = int(proc_h * scale)

            processed_preview = cv2.resize(processed_bgr, (new_w, new_h))

            # Create background
            bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = processed_preview

            # Add title and info
            cv2.putText(bg, "WHAT OCR SEES", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add image dimensions
            dim_text = f"Size: {proc_w}x{proc_h}"
            cv2.putText(bg, dim_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Copy to display - adjust position for larger preview
            display[0:400, 800:1400] = bg
            return bg

        except Exception as e:
            self.logger.error(f"Processed preview creation error: {e}")
            return None

    def _add_ocr_debug_info(self, display, frame):
        """Add debug information about what OCR is processing"""
        try:
            # Get score region info
            score_region = self.cfg.get_score_region()
            if not score_region:
                cv2.putText(display, "NO SCORE REGION CONFIGURED!", (900, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return

            # Extract score region for analysis
            score_image = self._extract_score_region(frame, score_region)
            if score_image is None:
                cv2.putText(display, "SCORE REGION EXTRACTION FAILED!", (900, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return

            # Show original score region info (top right)
            orig_h, orig_w = score_image.shape[:2]
            cv2.putText(display, f"Original: {orig_w}x{orig_h}", (900, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show score region coordinates
            x, y = int(score_region['x']), int(score_region['y'])
            w, h = int(score_region['width']), int(score_region['height'])
            cv2.putText(display, f"Coords: ({x},{y}) {w}x{h}", (900, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show frame dimensions for reference
            frame_h, frame_w = frame.shape[:2]
            cv2.putText(display, f"Frame: {frame_w}x{frame_h}", (900, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show processed image info (bottom right)
            processed = self.preprocess_image(score_image)
            proc_h, proc_w = processed.shape[:2]
            cv2.putText(display, f"Processed: {proc_w}x{proc_h}", (900, 650),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show preprocessing parameters
            cv2.putText(display, f"Brightness: {self.params['brightness']}", (900, 670),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display, f"Contrast: {self.params['contrast']:.1f}", (900, 685),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display, f"Sharpness: {self.params['sharpness']:.1f}", (900, 700),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display, f"Blur: {self.params['gaussian_blur']}", (900, 715),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show image statistics
            if len(processed.shape) == 2:  # Grayscale
                mean_val = np.mean(processed)
                std_val = np.std(processed)
                cv2.putText(display, f"Mean: {mean_val:.1f}", (900, 730),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display, f"Std: {std_val:.1f}", (900, 745),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Add histogram info
                hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
                min_val = np.min(processed)
                max_val = np.max(processed)
                cv2.putText(display, f"Range: {min_val}-{max_val}", (900, 760),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        except Exception as e:
            self.logger.error(f"OCR debug info error: {e}")
            cv2.putText(display, f"Debug Error: {str(e)[:30]}", (900, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _add_ocr_info(self, display, ocr_result, confidence):
        """Add OCR results and confidence to display"""
        try:
            # Large OCR result display
            result_text = f"SCORE: {ocr_result if ocr_result else 'NO DETECTION'}"
            cv2.putText(display, result_text, (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

            # Confidence display
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(display, conf_text, (10, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        except Exception as e:
            self.logger.error(f"OCR info display error: {e}")

    def _add_parameter_info(self, display):
        """Add current parameters to display"""
        try:
            param_text = (f"Brightness: {self.params['brightness']} | "
                         f"Contrast: {self.params['contrast']:.1f} | "
                         f"Sharpness: {self.params['sharpness']:.1f} | "
                         f"Blur: {self.params['gaussian_blur']} | "
                         f"Min Conf: {self.params['min_confidence']:.2f}")
            cv2.putText(display, param_text, (10, 550),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            self.logger.error(f"Parameter info display error: {e}")

    def _add_performance_info(self, display):
        """Add performance information to display"""
        try:
            # FPS calculation
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time

            # FPS display
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display, fps_text, (10, 580),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # OCR timing
            if self.ocr_times:
                avg_ocr_time = sum(self.ocr_times) / len(self.ocr_times)
                ocr_text = f"OCR: {avg_ocr_time*1000:.1f}ms"
                cv2.putText(display, ocr_text, (200, 580),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

            # GPU status
            gpu_status = "GPU: ON" if self.gpu_available else "GPU: OFF"
            cv2.putText(display, gpu_status, (10, 610),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if self.gpu_available else (128, 128, 128), 2)

        except Exception as e:
            self.logger.error(f"Performance info display error: {e}")

    def run_calibration(self):
        """Run the calibration interface"""
        self.create_calibration_window()
        self.running = True

        print("=== OCR CALIBRATION STARTED ===")
        print("Controls:")
        print("  Brightness: Adjust image brightness")
        print("  Contrast: Adjust image contrast")
        print("  Sharpness: Sharpen/blur the image")
        print("  Blur: Gaussian blur amount")
        print("  Min Confidence: Minimum OCR confidence threshold")
        print("Press 'S' to save configuration, 'Q' to quit")
        print("="*50)

        while self.running:
            frame = self.get_frame()

            if frame is not None and frame.size > 0:
                self.fps_counter += 1

                # Update parameters from sliders
                self.update_params()

                # Perform OCR
                ocr_result, confidence = self.perform_ocr(frame)

                # Create display frame
                display_frame = self.create_display_frame(frame, ocr_result, confidence)

                # Show frame
                cv2.imshow('OCR Calibration', display_frame)

            else:
                # No signal display
                no_signal = np.zeros((800, 1400, 3), dtype=np.uint8)  # Increased width
                cv2.putText(no_signal, "WAITING FOR SIGNAL", (400, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(no_signal, "Check capture area configuration", (350, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.imshow('OCR Calibration', no_signal)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord('s'):  # S
                self.save_optimal_config()
                print("✅ Configuration saved successfully!")

        # Cleanup
        self._cleanup()

    def save_optimal_config(self):
        """Save the current optimal configuration"""
        try:
            config = configparser.ConfigParser()
            config_path = Path(__file__).parent / "config" / "optimal_ocr_config.ini"

            config.add_section('OCR_OPTIMIZATION')
            for key, value in self.params.items():
                config.set('OCR_OPTIMIZATION', key, str(value))

            with open(config_path, 'w') as configfile:
                config.write(configfile)

            self.logger.info(f"Optimal OCR configuration saved to {config_path}")

            # Also save to main config
            main_config_path = self.cfg.config_file
            main_config = configparser.ConfigParser()
            main_config.read(main_config_path)

            if not main_config.has_section('OCR_OPTIMIZATION'):
                main_config.add_section('OCR_OPTIMIZATION')

            for key, value in self.params.items():
                main_config.set('OCR_OPTIMIZATION', key, str(value))

            with open(main_config_path, 'w') as configfile:
                main_config.write(configfile)

            self.logger.info("Configuration also saved to main config file")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def _cleanup(self):
        """Clean up resources"""
        try:
            self.running = False
            if hasattr(self, 'cap'):
                self.cap.stop()

            cv2.destroyAllWindows()
            self.logger.info("OCR Calibration cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


def main():
    """Main function to run the OCR calibrator"""
    try:
        calibrator = OcrCalibrator()
        calibrator.run_calibration()

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    except Exception as e:
        print(f"\nError during calibration: {e}")
        logging.exception("Calibration error")


if __name__ == "__main__":
    main()
