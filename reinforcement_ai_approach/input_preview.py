import time
import threading
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging
import torch
import easyocr
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import setup_logger
    from src.core.screen_capture import ScreenCapture
    from src.core.score_detector import ScoreDetector
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    sys.exit(1)

# Check EasyOCR availability
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("WARNING: EasyOCR not available. Install with: pip install easyocr")


class DisplayLayer(Enum):
    """Enum for display layers to avoid overlapping"""
    BACKGROUND = 0
    GAME_FRAME = 1
    LANE_OVERLAY = 2
    SCORE_REGION = 3
    OCR_DEBUG = 4
    PERFORMANCE_INFO = 5
    UI_ELEMENTS = 6


@dataclass
class VisualElement:
    """Represents a visual element with position and priority"""
    layer: DisplayLayer
    x: int
    y: int
    width: int
    height: int
    content: np.ndarray
    priority: int = 0


class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.fps_history = []
        self.ocr_times = []
        self.frame_times = []
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def update_frame(self):
        """Update frame timing"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
        self.last_frame_time = current_time
        self.frame_count += 1
        
    def update_ocr_time(self, ocr_time: float):
        """Update OCR timing"""
        self.ocr_times.append(ocr_time)
        if len(self.ocr_times) > self.window_size:
            self.ocr_times.pop(0)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_avg_ocr_time(self) -> float:
        """Get average OCR time"""
        if not self.ocr_times:
            return 0.0
        return sum(self.ocr_times) / len(self.ocr_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance stats"""
        return {
            'fps': self.get_fps(),
            'frame_count': self.frame_count,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0.0,
            'avg_ocr_time': self.get_avg_ocr_time(),
            'min_ocr_time': min(self.ocr_times) if self.ocr_times else 0.0,
            'max_ocr_time': max(self.ocr_times) if self.ocr_times else 0.0
        }


class OptimizedOCRProcessor:
    """Professional OCR processor with GPU acceleration and optimized performance"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = setup_logger(__name__)
        self.reader = None
        self.params = self._load_ocr_params()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize OCR engine
        self._initialize_ocr_engine()
        
    def _load_ocr_params(self) -> Dict[str, Any]:
        """Load optimized OCR parameters from config"""
        def safe_get(section: str, key: str, default: Any) -> Any:
            """Safely get config value with default fallback"""
            try:
                if section == 'OCR_OPTIMIZATION' and key == 'min_confidence':
                    return self.config.getint(section, key) / 100.0
                elif key in ['brightness', 'contrast', 'sharpness', 'min_digits', 'max_digits']:
                    return self.config.getint(section, key)
                elif key in ['max_score_change', 'min_score_change']:
                    return self.config.getint(section, key)
                else:
                    return self.config.get(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
                return default
        
        return {
            'brightness': safe_get('OCR_OPTIMIZATION', 'brightness', 0),
            'contrast': safe_get('OCR_OPTIMIZATION', 'contrast', 5),
            'sharpness': safe_get('OCR_OPTIMIZATION', 'sharpness', 11),
            'min_confidence': safe_get('OCR_OPTIMIZATION', 'min_confidence', 0.8),
            'min_digits': safe_get('OCR_OPTIMIZATION', 'min_digits', 1),
            'max_digits': safe_get('OCR_OPTIMIZATION', 'max_digits', 8),
            'max_score_change': safe_get('OCR_OPTIMIZATION', 'max_score_change', 50000),
            'min_score_change': safe_get('OCR_OPTIMIZATION', 'min_score_change', -1000),
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU detected: {gpu_name}")
                self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
                return True
            else:
                self.logger.info("No GPU detected - using CPU mode")
                return False
        except Exception as e:
            self.logger.warning(f"Error checking GPU availability: {e}")
            return False
    
    def _initialize_ocr_engine(self):
        """Initialize EasyOCR with optimized settings"""
        if not EASYOCR_AVAILABLE:
            self.logger.error("EasyOCR not available. Please install with: pip install easyocr")
            return
            
        try:
            gpu_enabled = self._check_gpu_availability()
            reader_kwargs = {
                'lang_list': ['en'],
                'gpu': gpu_enabled,
                'model_storage_directory': None,
                'user_network_directory': None,
                'recognizer': True,
                'detector': True,
                'verbose': False,
            }
            
            self.logger.info(f"Initializing EasyOCR with GPU: {gpu_enabled}")
            self.reader = easyocr.Reader(**reader_kwargs)
            
            # Warm up the engine
            self._warm_up_engine()
            
            self.logger.info("✅ EasyOCR initialized successfully with GPU acceleration")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def _warm_up_engine(self):
        """Perform dummy inference for better performance"""
        try:
            if self.reader:
                dummy_image = np.zeros((50, 200), dtype=np.uint8)
                self.reader.readtext(dummy_image)
                self.logger.info("OCR engine warm-up completed")
        except Exception as e:
            self.logger.warning(f"Warm-up failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply optimized preprocessing to image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply brightness and contrast
            if self.params['brightness'] != 0 or self.params['contrast'] != 1.0:
                gray = cv2.add(gray, self.params['brightness'])
                gray = cv2.multiply(gray, self.params['contrast'])
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            
            # Apply sharpening if needed
            if self.params['sharpness'] > 0:
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]]) * self.params['sharpness'] / 10.0
                gray = cv2.filter2D(gray, -1, kernel)
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            
            return gray
            
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            return image
    
    def extract_score_region(self, frame: np.ndarray, score_region: Dict[str, int]) -> Optional[np.ndarray]:
        """Extract the score region from the frame"""
        try:
            x, y = int(score_region['x']), int(score_region['y'])
            w, h = int(score_region['width']), int(score_region['height'])
            
            # Handle relative coordinates
            if (x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]):
                return frame[y:y+h, x:x+w]
            else:
                # Try absolute to relative conversion
                capture_config = self.config.get_capture_area_config()
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
            self.logger.error(f"Score region extraction error: {e}")
            return None
    
    def perform_ocr(self, frame: np.ndarray) -> Tuple[str, float]:
        """Perform OCR on the frame with optimized processing"""
        if not self.reader:
            return "OCR not initialized", 0.0
            
        try:
            # Get score region
            score_region = self.config.get_score_region()
            if not score_region:
                return "No score region configured", 0.0
            
            # Extract score region
            score_image = self.extract_score_region(frame, score_region)
            if score_image is None:
                return "Invalid score region", 0.0
            
            # Preprocess image
            processed_image = self.preprocess_image(score_image)
            
            # Perform OCR with timing
            start_time = time.time()
            results = self.reader.readtext(processed_image)
            ocr_time = time.time() - start_time
            
            # Update performance monitor
            self.performance_monitor.update_ocr_time(ocr_time)
            
            # Process results
            best_text = ""
            best_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                if confidence > self.params['min_confidence']:
                    # Extract only digits
                    cleaned = ''.join(filter(str.isdigit, text))
                    if len(cleaned) >= self.params['min_digits'] and confidence > best_confidence:
                        best_text = cleaned
                        best_confidence = confidence
            
            return best_text, best_confidence
            
        except Exception as e:
            self.logger.error(f"OCR processing error: {e}")
            return "", 0.0
    
    def get_processed_image(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get the processed image that OCR sees"""
        try:
            score_region = self.config.get_score_region()
            if not score_region:
                return None
            
            score_image = self.extract_score_region(frame, score_region)
            if score_image is None:
                return None
            
            return self.preprocess_image(score_image)
            
        except Exception as e:
            self.logger.error(f"Error getting processed image: {e}")
            return None


class LaneProcessor:
    """Handles lane detection with optimized processing"""
    
    def __init__(self, max_workers: int = 6):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = setup_logger(__name__)
        
    def process_lanes_parallel(self, frame: np.ndarray, polygons: Dict[str, List[Tuple[int, int]]], 
                             hsv_ranges: Dict[str, Dict], morph: Dict[str, int]) -> Dict[str, Dict[str, bool]]:
        """Process all lanes in parallel for maximum performance"""
        try:
            # Convert to HSV once
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Submit all lane processing tasks
            futures = {}
            for lane_name, polygon in polygons.items():
                if not polygon:
                    continue
                    
                # Create mask for this lane
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
                
                future = self.executor.submit(
                    self._process_single_lane, lane_name, mask, hsv_frame, hsv_ranges, morph
                )
                futures[future] = lane_name
            
            # Collect results
            results = {}
            for future in as_completed(futures, timeout=0.1):
                try:
                    result = future.result()
                    results[result['lane_name']] = result
                except Exception as e:
                    self.logger.error(f"Error processing lane: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lane processing error: {e}")
            return {}
    
    def _process_single_lane(self, lane_name: str, mask: np.ndarray, hsv_frame: np.ndarray,
                           hsv_ranges: Dict[str, Dict], morph: Dict[str, int]) -> Dict[str, Any]:
        """Process a single lane"""
        try:
            # Detect colors
            g_mask = self._detect_color_mask(hsv_frame, 'green', hsv_ranges, morph)
            y_mask = self._detect_color_mask(hsv_frame, 'yellow', hsv_ranges, morph)
            
            # Apply lane mask
            g_present = cv2.countNonZero(cv2.bitwise_and(g_mask, mask)) > 0
            y_present = cv2.countNonZero(cv2.bitwise_and(y_mask, mask)) > 0
            
            return {
                'lane_name': lane_name,
                'g_present': g_present,
                'y_present': y_present
            }
            
        except Exception as e:
            self.logger.error(f"Single lane processing error for {lane_name}: {e}")
            return {
                'lane_name': lane_name,
                'g_present': False,
                'y_present': False
            }
    
    def _detect_color_mask(self, hsv_img: np.ndarray, color: str, 
                          hsv_ranges: Dict[str, Dict], morph: Dict[str, int]) -> np.ndarray:
        """Detect color mask with morphology"""
        try:
            lower = np.array([hsv_ranges[color]['h_min'],
                             hsv_ranges[color]['s_min'],
                             hsv_ranges[color]['v_min']])
            upper = np.array([hsv_ranges[color]['h_max'],
                             hsv_ranges[color]['s_max'],
                             hsv_ranges[color]['v_max']])
            
            mask = cv2.inRange(hsv_img, lower, upper)
            
            # Apply morphology
            close_size = morph['close_size']
            dilate_size = morph['dilate_size']
            
            if color == 'yellow':
                close_size = max(3, close_size // 2)
                dilate_size = max(2, dilate_size // 2)
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                                   np.ones((close_size, close_size), np.uint8))
            mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Mask detection error for {color}: {e}")
            return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


class VisualRenderer:
    """Handles all visual rendering with organized layers"""
    
    def __init__(self, display_width: int = 1600, display_height: int = 900):
        self.display_width = display_width
        self.display_height = display_height
        self.logger = setup_logger(__name__)
        
        # Define layout regions
        self.game_region = (0, 0, 1200, 800)  # Main game area
        self.info_panel = (1200, 0, 400, 800)  # Right side info panel
        self.ocr_debug = (0, 800, 400, 100)    # Bottom OCR debug area (half width)
        
    def create_display_canvas(self) -> np.ndarray:
        """Create the main display canvas"""
        return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
    
    def render_game_frame(self, frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """Render the main game frame"""
        try:
            x, y, w, h = self.game_region
            # Resize frame to fit game region
            game_frame = cv2.resize(frame, (w, h))
            canvas[y:y+h, x:x+w] = game_frame
            return canvas
        except Exception as e:
            self.logger.error(f"Game frame rendering error: {e}")
            return canvas
    
    def render_lane_overlay(self, canvas: np.ndarray, polygons: Dict[str, List[Tuple[int, int]]], 
                           lane_results: Dict[str, Dict[str, bool]]) -> np.ndarray:
        """Render lane detection overlay"""
        try:
            x, y, w, h = self.game_region
            scale_x = w / 1920  # Assuming original frame is 1920x1080
            scale_y = h / 1080
            
            for lane_name, lane_result in lane_results.items():
                if lane_name not in polygons or not polygons[lane_name]:
                    continue
                
                # Scale polygon points
                scaled_polygon = [(int(px * scale_x), int(py * scale_y)) for px, py in polygons[lane_name]]
                pts = np.array(scaled_polygon, np.int32)
                
                # Choose color based on detection
                if lane_result['g_present']:
                    color = (0, 255, 0)  # Green
                    fill_color = (0, 120, 0)
                elif lane_result['y_present']:
                    color = (0, 255, 255)  # Yellow
                    fill_color = (0, 120, 120)
                else:
                    color = (120, 120, 120)  # Gray
                    fill_color = None
                
                # Draw lane outline
                cv2.polylines(canvas, [pts], True, color, 2)
                
                # Fill lane if note detected
                if fill_color:
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts], fill_color)
                    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
                
                # Add lane info
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(canvas, f"{lane_name} G:{int(lane_result['g_present'])} Y:{int(lane_result['y_present'])}",
                           tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Lane overlay rendering error: {e}")
            return canvas
    
    def render_score_region(self, canvas: np.ndarray, score_region: Dict[str, int]) -> np.ndarray:
        """Render score region highlight"""
        try:
            if not score_region:
                return canvas
                
            x, y, w, h = self.game_region
            # Use the actual score region coordinates from config
            # The score region is already in the correct coordinate system
            score_x = int(score_region['x'])
            score_y = int(score_region['y'])
            score_w = int(score_region['width'])
            score_h = int(score_region['height'])
            
            # Scale to fit the game region (assuming original frame is 1920x1080)
            scale_x = w / 1920
            scale_y = h / 1080
            
            # Scale score region coordinates
            scaled_x = int(score_x * scale_x)
            scaled_y = int(score_y * scale_y)
            scaled_w = int(score_w * scale_x)
            scaled_h = int(score_h * scale_y)
            
            # Draw score region
            cv2.rectangle(canvas, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (255, 255, 0), 2)
            cv2.putText(canvas, "SCORE", (scaled_x, scaled_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Score region rendering error: {e}")
            return canvas
    
    def render_info_panel(self, canvas: np.ndarray, current_score: int, ocr_result: str, 
                         ocr_confidence: float, perf_stats: Dict[str, Any]) -> np.ndarray:
        """Render information panel on the right side"""
        try:
            x, y, w, h = self.info_panel
            
            # Create info panel background
            info_bg = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Score information
            cv2.putText(info_bg, "SCORE INFO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_bg, f"Current: {current_score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # OCR information
            cv2.putText(info_bg, "OCR STATUS", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if ocr_result:
                cv2.putText(info_bg, f"Detected: {ocr_result}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(info_bg, f"Confidence: {ocr_confidence:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(info_bg, "No score detected", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Performance information
            cv2.putText(info_bg, "PERFORMANCE", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_bg, f"FPS: {perf_stats['fps']:.1f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(info_bg, f"OCR: {perf_stats['avg_ocr_time']*1000:.1f}ms", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            cv2.putText(info_bg, f"Frames: {perf_stats['frame_count']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Copy to canvas
            canvas[y:y+h, x:x+w] = info_bg
            return canvas
            
        except Exception as e:
            self.logger.error(f"Info panel rendering error: {e}")
            return canvas
    
    def render_ocr_debug(self, canvas: np.ndarray, processed_img: Optional[np.ndarray]) -> np.ndarray:
        """Render OCR debug information at the bottom"""
        try:
            x, y, w, h = self.ocr_debug
            
            # Create debug area background
            debug_bg = np.zeros((h, w, 3), dtype=np.uint8)
            
            if processed_img is not None and processed_img.size > 0:
                try:
                    # Resize processed image to fit debug area
                    debug_img = cv2.resize(processed_img, (w-20, h-20))
                    
                    # Convert to BGR if grayscale
                    if len(debug_img.shape) == 2:
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                    
                    # Invert if mostly black for better visibility
                    # Convert to grayscale for countNonZero
                    if len(debug_img.shape) == 3:
                        gray_debug = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_debug = debug_img
                    
                    if cv2.countNonZero(gray_debug) < (gray_debug.shape[0] * gray_debug.shape[1] * 0.1):
                        debug_img = cv2.bitwise_not(debug_img)
                    
                    # Add title
                    cv2.putText(debug_bg, "OCR Input Preview", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Place image safely
                    y_start = 30
                    y_end = min(y_start + debug_img.shape[0], h)
                    x_start = 10
                    x_end = min(x_start + debug_img.shape[1], w)
                    
                    # Ensure we don't exceed boundaries
                    if y_end > y_start and x_end > x_start:
                        debug_bg[y_start:y_end, x_start:x_end] = debug_img[:y_end-y_start, :x_end-x_start]
                    
                    # Add image info
                    img_h, img_w = processed_img.shape[:2]
                    cv2.putText(debug_bg, f"Size: {img_w}x{img_h}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                except Exception as e:
                    # If image placement fails, just show error
                    cv2.putText(debug_bg, f"Image Error: {str(e)[:20]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(debug_bg, "No OCR input available", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Copy to canvas
            canvas[y:y+h, x:x+w] = debug_bg
            return canvas
            
        except Exception as e:
            self.logger.error(f"OCR debug rendering error: {e}")
            return canvas


class OptimizedPreviewSystem:
    """Professional preview system with optimized architecture"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.ocr_processor = OptimizedOCRProcessor(config)
        self.lane_processor = LaneProcessor()
        self.visual_renderer = VisualRenderer()
        self.performance_monitor = PerformanceMonitor()
        
        # State management
        self.current_score = 0
        self.score_lock = threading.Lock()
        self.running = False
        
    def process_frame(self, frame: np.ndarray, polygons: Dict[str, List[Tuple[int, int]]], 
                     hsv_ranges: Dict[str, Dict], morph: Dict[str, int]) -> Dict[str, Any]:
        """Process frame with all optimizations"""
        try:
            # Update performance monitoring
            self.performance_monitor.update_frame()
            
            # Process lanes in parallel
            lane_results = self.lane_processor.process_lanes_parallel(frame, polygons, hsv_ranges, morph)
            
            # Process OCR
            ocr_result, confidence = self.ocr_processor.perform_ocr(frame)
            
            # Update score if valid
            if ocr_result and ocr_result.isdigit():
                try:
                    new_score = int(ocr_result)
                    with self.score_lock:
                        self.current_score = new_score
                except ValueError:
                    pass
            
            # Get performance stats
            perf_stats = self.performance_monitor.get_stats()
            
            return {
                'lane_results': lane_results,
                'ocr_result': ocr_result,
                'ocr_confidence': confidence,
                'current_score': self.current_score,
                'perf_stats': perf_stats
            }
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return {
                'lane_results': {},
                'ocr_result': "",
                'ocr_confidence': 0.0,
                'current_score': self.current_score,
                'perf_stats': self.performance_monitor.get_stats()
            }
    
    def render_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Render the complete frame with all visual elements"""
        try:
            # Create display canvas
            canvas = self.visual_renderer.create_display_canvas()
            
            # Render game frame
            canvas = self.visual_renderer.render_game_frame(frame, canvas)
            
            # Render lane overlay
            canvas = self.visual_renderer.render_lane_overlay(canvas, 
                self.config.get_note_lane_polygons_relative(), results['lane_results'])
            
            # Render score region
            canvas = self.visual_renderer.render_score_region(canvas, 
                self.config.get_score_region())
            
            # Render info panel
            canvas = self.visual_renderer.render_info_panel(canvas, 
                results['current_score'], results['ocr_result'], 
                results['ocr_confidence'], results['perf_stats'])
            
            # Render OCR debug
            processed_img = self.ocr_processor.get_processed_image(frame)
            canvas = self.visual_renderer.render_ocr_debug(canvas, processed_img)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Frame rendering error: {e}")
            # Return error frame
            error_canvas = np.zeros((900, 1600, 3), dtype=np.uint8)
            cv2.putText(error_canvas, "Rendering Error", (700, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return error_canvas
    
    def cleanup(self):
        """Clean up all resources"""
        try:
            self.lane_processor.cleanup()
            self.logger.info("Preview system cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


def main():
    """Main function with professional error handling and optimized architecture"""
    try:
        # Initialize configuration
        cfg_path = Path(__file__).parent / 'config' / 'config.ini'
        cfg = ConfigManager(config_path=str(cfg_path))
        
        # Get configuration parameters
        hsv_ranges = cfg.get_hsv_ranges()
        morph = cfg.get_morphology_params()
        polygons = cfg.get_note_lane_polygons_relative()
        
        # Initialize optimized system
        preview_system = OptimizedPreviewSystem(cfg)
        
        # Initialize screen capture
        cap = ScreenCapture(cfg)
        cap.start()
        
        # Wait for capture to initialize
        time.sleep(0.5)
        
        # Create window
        cv2.namedWindow('Input Preview - Professional', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Input Preview - Professional', 1600, 900)
        
        # Main loop
        try:
            while True:
                frame = cap.get_latest_frame()
                if frame is None or frame.size == 0:
                    cv2.waitKey(1)
                    continue
                
                # Process frame with all optimizations
                results = preview_system.process_frame(frame, polygons, hsv_ranges, morph)
                
                # Render complete frame
                display_frame = preview_system.render_frame(frame, results)
                
                # Display frame
                cv2.imshow('Input Preview - Professional', display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('i'):  # I for info
                    print(f"\n=== PERFORMANCE INFO ===")
                    perf_stats = results['perf_stats']
                    print(f"FPS: {perf_stats['fps']:.1f}")
                    print(f"Frame Count: {perf_stats['frame_count']}")
                    print(f"Avg Frame Time: {perf_stats['avg_frame_time']*1000:.1f}ms")
                    print(f"Avg OCR Time: {perf_stats['avg_ocr_time']*1000:.1f}ms")
                    print(f"Min OCR Time: {perf_stats['min_ocr_time']*1000:.1f}ms")
                    print(f"Max OCR Time: {perf_stats['max_ocr_time']*1000:.1f}ms")
                    print(f"Current Score: {results['current_score']}")
                    print(f"OCR Result: {results['ocr_result']}")
                    print(f"OCR Confidence: {results['ocr_confidence']:.2f}")
                    print("========================")
        
        finally:
            # Cleanup
            cap.stop()
            preview_system.cleanup()
            cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("\nPreview interrupted by user")
    except Exception as e:
        print(f"\nError during preview: {e}")
        logging.exception("Preview error")


if __name__ == '__main__':
    main()