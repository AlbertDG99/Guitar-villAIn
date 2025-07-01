#!/usr/bin/env python3
"""
🚀 CUDA Experimental Detector - OpenCV GPU Acceleration
=========================================================

Basado en documentación oficial de OpenCV CUDA:
https://opencv.org/platforms/cuda/

Implementa:
- cv::cuda::GpuMat para datos GPU
- Operaciones CUDA nativas donde sea posible
- Fallback automático a CPU
- Métricas de rendimiento CPU vs GPU
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

from core.screen_capture import ScreenCapture
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

@dataclass
class CudaDetection:
    """Resultado de detección con información CUDA"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    color_type: str
    lane: str
    processing_time: float
    used_gpu: bool

class CudaExperimentalDetector:
    """
    Detector experimental que aprovecha CUDA de OpenCV donde sea posible.
    
    Implementa las mejores prácticas de OpenCV CUDA:
    - GpuMat para datos en GPU
    - Operaciones CUDA nativas
    - Gestión eficiente de memoria GPU
    - Pipeline optimizado GPU->CPU
    """
    
    def __init__(self, confidence_threshold: float = 0.95, max_workers: int = 6):
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager()
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        
        # Inicialización CUDA
        self.cuda_available = self._initialize_cuda()
        self.gpu_stream = None
        if self.cuda_available:
            self.gpu_stream = cv2.cuda.Stream()
            
        # Configuración de detección
        self.screen_capture = ScreenCapture(self.config_manager)
        self.lane_polygons = self._load_lane_polygons()
        self.color_ranges = self._load_color_ranges()
        
        # Métricas
        self.detection_stats = {
            'gpu_operations': 0,
            'cpu_operations': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'total_detections': 0
        }
        
        self.logger.info(f"🚀 CUDA Experimental Detector iniciado")
        self.logger.info(f"   🎯 Threshold: {confidence_threshold}")
        self.logger.info(f"   🖥️ CUDA disponible: {self.cuda_available}")
        self.logger.info(f"   🧵 Workers: {max_workers}")

    def _initialize_cuda(self) -> bool:
        """Inicializa CUDA siguiendo mejores prácticas de OpenCV"""
        try:
            if not hasattr(cv2, 'cuda'):
                self.logger.warning("❌ OpenCV sin módulo CUDA")
                return False
                
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count == 0:
                self.logger.warning("❌ No se detectaron dispositivos CUDA")
                return False
                
            # Configurar dispositivo CUDA
            cv2.cuda.setDevice(0)
            
            # Información del dispositivo
            device_info = cv2.cuda.DeviceInfo(0)
            self.logger.info(f"✅ CUDA Device: {device_info.name()}")
            self.logger.info(f"   💾 Global Memory: {device_info.totalGlobalMem() / (1024**3):.1f} GB")
            self.logger.info(f"   🔢 Compute Capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error inicializando CUDA: {e}")
            return False

    def _load_lane_polygons(self) -> Dict[str, List[Tuple[int, int]]]:
        """Carga polígonos de carriles y los convierte a coordenadas relativas"""
        lane_keys = ['s', 'd', 'f', 'j', 'k', 'l']
        lane_map = {'s': 'S', 'd': 'D', 'f': 'F', 'j': 'J', 'k': 'K', 'l': 'L'}
        
        polygons = {}
        game_left = int(self.config_manager.get('CAPTURE', 'game_left', '0'))
        game_top = int(self.config_manager.get('CAPTURE', 'game_top', '0'))
        
        for key in lane_keys:
            section = f'NOTE_LANE_{lane_map[key]}'
            if self.config_manager.config.has_section(section):
                points = []
                for i in range(1, 5):
                    x = int(self.config_manager.get(section, f'point_{i}_x', '0'))
                    y = int(self.config_manager.get(section, f'point_{i}_y', '0'))
                    # Convertir a coordenadas relativas
                    rel_x = x - game_left
                    rel_y = y - game_top
                    points.append((rel_x, rel_y))
                polygons[key] = points
                
        self.logger.info(f"✅ Polígonos cargados para teclas: {list(polygons.keys())}")
        return polygons

    def _load_color_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Carga rangos HSV optimizados para cada color"""
        ranges = {
            'yellow': (
                np.array([15, 100, 100], dtype=np.uint8),
                np.array([40, 255, 255], dtype=np.uint8)
            ),
            'green': (
                np.array([25, 40, 40], dtype=np.uint8),
                np.array([95, 255, 255], dtype=np.uint8)
            )
        }
        return ranges

    def detect_frame_cuda(self, frame: np.ndarray) -> List[CudaDetection]:
        """
        Detección principal usando CUDA donde sea posible
        
        Implementa pipeline optimizado:
        1. Upload to GPU (cv::cuda::GpuMat)
        2. GPU operations when possible
        3. Download results
        4. Fallback to CPU for unsupported operations
        """
        start_time = time.time()
        detections = []
        
        if self.cuda_available:
            detections = self._detect_with_gpu(frame)
        else:
            detections = self._detect_with_cpu(frame)
            
        total_time = (time.time() - start_time) * 1000
        self.detection_stats['total_detections'] += len(detections)
        
        return detections

    def _detect_with_gpu(self, frame: np.ndarray) -> List[CudaDetection]:
        """Detección usando operaciones CUDA donde sea posible"""
        gpu_start = time.time()
        detections = []
        
        try:
            # 1. Upload frame to GPU (según documentación OpenCV CUDA)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # 2. Convert to HSV on GPU
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV, stream=self.gpu_stream)
            
            # 3. Process each lane in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for lane, polygon in self.lane_polygons.items():
                    future = executor.submit(self._process_lane_gpu, gpu_hsv, lane, polygon)
                    futures.append(future)
                
                for future in as_completed(futures):
                    lane_detections = future.result()
                    detections.extend(lane_detections)
            
            # Esperar a que todas las operaciones GPU terminen
            if self.gpu_stream:
                self.gpu_stream.waitForCompletion()
                
            gpu_time = (time.time() - gpu_start) * 1000
            self.detection_stats['gpu_operations'] += 1
            self.detection_stats['gpu_time'] += gpu_time
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU operation failed, fallback to CPU: {e}")
            return self._detect_with_cpu(frame)

    def _process_lane_gpu(self, gpu_hsv: Any, lane: str, polygon: List[Tuple[int, int]]) -> List[CudaDetection]:
        """Procesa un carril usando operaciones GPU donde sea posible"""
        lane_start = time.time()
        detections = []
        
        try:
            # Crear máscara del polígono (CPU - no disponible en CUDA)
            height, width = gpu_hsv.size()
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], (255,))
            
            # Upload mask to GPU
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(mask)
            
            # Aplicar máscara en GPU
            gpu_roi = cv2.cuda.bitwise_and(gpu_hsv, gpu_hsv, mask=gpu_mask, stream=self.gpu_stream)
            
            # Detectar cada color
            for color_name, (lower, upper) in self.color_ranges.items():
                color_detections = self._detect_color_gpu(gpu_roi, color_name, lower, upper, lane)
                detections.extend(color_detections)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Lane GPU processing failed for {lane}: {e}")
            # Fallback a CPU para este carril
            hsv_cpu = gpu_hsv.download()
            detections = self._process_lane_cpu(hsv_cpu, lane, polygon)
        
        lane_time = (time.time() - lane_start) * 1000
        
        # Marcar detecciones como procesadas por GPU
        for detection in detections:
            detection.processing_time = lane_time
            detection.used_gpu = True
            
        return detections

    def _detect_color_gpu(self, gpu_roi: Any, color_name: str, lower: np.ndarray, 
                         upper: np.ndarray, lane: str) -> List[CudaDetection]:
        """Detecta un color específico usando operaciones GPU"""
        detections = []
        
        try:
            # Crear máscara de color en GPU
            gpu_mask = cv2.cuda.inRange(gpu_roi, lower, upper, stream=self.gpu_stream)
            
            # Operaciones morfológicas en GPU (si están disponibles)
            if hasattr(cv2.cuda, 'morphologyEx'):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                gpu_kernel = cv2.cuda_GpuMat()
                gpu_kernel.upload(kernel)
                gpu_mask = cv2.cuda.morphologyEx(gpu_mask, cv2.MORPH_CLOSE, gpu_kernel, stream=self.gpu_stream)
            
            # Download para análisis de contornos (no disponible en CUDA)
            mask_cpu = gpu_mask.download()
            
            # Encontrar contornos (CPU only)
            contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                confidence = self._calculate_confidence(contour, mask_cpu)
                threshold = self.confidence_threshold if color_name == 'yellow' else 0.85
                
                if confidence >= threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = CudaDetection(
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        color_type=color_name,
                        lane=lane,
                        processing_time=0,  # Se asignará en el padre
                        used_gpu=True
                    )
                    detections.append(detection)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ GPU color detection failed for {color_name}: {e}")
            
        return detections

    def _detect_with_cpu(self, frame: np.ndarray) -> List[CudaDetection]:
        """Fallback a detección CPU cuando CUDA no está disponible"""
        cpu_start = time.time()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for lane, polygon in self.lane_polygons.items():
                future = executor.submit(self._process_lane_cpu, hsv, lane, polygon)
                futures.append(future)
            
            for future in as_completed(futures):
                lane_detections = future.result()
                detections.extend(lane_detections)
        
        cpu_time = (time.time() - cpu_start) * 1000
        self.detection_stats['cpu_operations'] += 1
        self.detection_stats['cpu_time'] += cpu_time
        
        return detections

    def _process_lane_cpu(self, hsv: np.ndarray, lane: str, polygon: List[Tuple[int, int]]) -> List[CudaDetection]:
        """Procesa un carril usando CPU"""
        lane_start = time.time()
        detections = []
        
        # Crear máscara del polígono
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255,))
        
        # Aplicar máscara
        roi = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        for color_name, (lower, upper) in self.color_ranges.items():
            color_mask = cv2.inRange(roi, lower, upper)
            
            # Operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                confidence = self._calculate_confidence(contour, color_mask)
                threshold = self.confidence_threshold if color_name == 'yellow' else 0.85
                
                if confidence >= threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = CudaDetection(
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        color_type=color_name,
                        lane=lane,
                        processing_time=(time.time() - lane_start) * 1000,
                        used_gpu=False
                    )
                    detections.append(detection)
        
        return detections

    def _calculate_confidence(self, contour: np.ndarray, mask: np.ndarray) -> float:
        """Calcula confianza usando algoritmo multi-factor"""
        area = cv2.contourArea(contour)
        if area < 10:
            return 0.0
            
        # Factor 1: Área normalizada
        area_factor = min(area / 500.0, 1.0)
        
        # Factor 2: Relación de aspecto
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        aspect_factor = 1.0 if 0.5 <= aspect_ratio <= 7.0 else 0.8
        
        # Factor 3: Solidez
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_factor = min(solidity * 1.3, 1.0)
        
        # Factor 4: Intensidad promedio
        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [contour], (255,))
        combined_mask = cv2.bitwise_and(mask, roi_mask)
        intensity = np.mean(combined_mask[combined_mask > 0]) if np.any(combined_mask > 0) else 0
        intensity_factor = min(intensity / 140.0, 1.0)
        
        # Combinación ponderada
        confidence = (
            area_factor * 0.30 +
            aspect_factor * 0.15 +
            solidity_factor * 0.35 +
            intensity_factor * 0.20
        )
        
        return confidence

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de rendimiento GPU vs CPU"""
        total_ops = self.detection_stats['gpu_operations'] + self.detection_stats['cpu_operations']
        if total_ops == 0:
            return {}
            
        avg_gpu_time = self.detection_stats['gpu_time'] / max(self.detection_stats['gpu_operations'], 1)
        avg_cpu_time = self.detection_stats['cpu_time'] / max(self.detection_stats['cpu_operations'], 1)
        
        return {
            'cuda_available': self.cuda_available,
            'total_operations': total_ops,
            'gpu_operations': self.detection_stats['gpu_operations'],
            'cpu_operations': self.detection_stats['cpu_operations'],
            'avg_gpu_time_ms': avg_gpu_time,
            'avg_cpu_time_ms': avg_cpu_time,
            'gpu_speedup': avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0,
            'total_detections': self.detection_stats['total_detections']
        }

    def cleanup(self):
        """Limpia recursos CUDA"""
        if self.cuda_available and self.gpu_stream:
            del self.gpu_stream
        self.logger.info("🧹 Recursos CUDA limpiados") 