#!/usr/bin/env python3
"""Polygon Calibrator - Ajustar y reducir polígonos de carriles"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.utils.config_manager import ConfigManager
from src.core.screen_capture import ScreenCapture

class PolygonCalibrator:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.screen_capture = ScreenCapture(self.config_manager)
        
        # Cargar polígonos existentes
        self.lane_polygons = self._load_existing_polygons()
        self.key_names = ['S', 'D', 'F', 'J', 'K', 'L']
        
        self.lane_colors = [
            (0, 255, 0), (0, 255, 255), (255, 0, 0),
            (255, 0, 255), (0, 128, 255), (128, 255, 0)
        ]
        
        # Estado de calibración
        self.current_lane = 0
        self.current_points = []
        self.completed_polygons = {}
        
        # Copiar polígonos existentes como base
        for lane_name, points in self.lane_polygons.items():
            self.completed_polygons[lane_name] = points.copy()
        
        self.screen_capture.initialize_mss()
        
        print("🎯 CALIBRADOR DE POLÍGONOS - Reducir tamaño para rendimiento")
        print("="*60)
        print("📊 Polígonos actuales:")
        self._show_current_polygons()
        
    def _load_existing_polygons(self) -> Dict[str, List[Tuple[int, int]]]:
        """Cargar polígonos existentes"""
        try:
            polygons = self.config_manager.get_note_lane_polygons()
            if polygons:
                print(f"✅ Cargados {len(polygons)} polígonos existentes")
                return polygons
            else:
                print("❌ No hay polígonos existentes")
                return {}
        except Exception as e:
            print(f"❌ Error cargando polígonos: {e}")
            return {}
    
    def _show_current_polygons(self):
        """Mostrar información de polígonos actuales"""
        for lane_name, points in self.lane_polygons.items():
            if points:
                # Calcular área del polígono
                pts = np.array(points, np.int32)
                area = cv2.contourArea(pts)
                
                # Calcular bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                print(f"  🔹 {lane_name}: {len(points)} puntos, área: {area:.0f}px², "
                      f"tamaño: {width}x{height}px")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Agregar punto
            self.current_points.append((x, y))
            print(f"  ➕ Punto {len(self.current_points)}: ({x}, {y})")
            
            # Si tenemos 4 puntos, completar polígono
            if len(self.current_points) >= 4:
                lane_name = self.key_names[self.current_lane]
                self.completed_polygons[lane_name] = self.current_points.copy()
                
                # Calcular área del nuevo polígono
                pts = np.array(self.current_points, np.int32)
                area = cv2.contourArea(pts)
                
                print(f"✅ Polígono {lane_name} completado: {area:.0f}px²")
                
                # Pasar al siguiente carril
                self.current_lane += 1
                self.current_points = []
                
                if self.current_lane >= len(self.key_names):
                    print("\n🎉 ¡Todos los polígonos calibrados!")
                    print("Presiona 's' para guardar o 'r' para reiniciar")
                else:
                    next_lane = self.key_names[self.current_lane]
                    print(f"\n🎯 Ahora calibra el carril {next_lane} ({self.current_lane + 1}/{len(self.key_names)})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Deshacer último punto
            if self.current_points:
                removed = self.current_points.pop()
                print(f"  ↩️ Punto eliminado: {removed}")
    
    def draw_polygons(self, frame: np.ndarray) -> np.ndarray:
        """Dibujar polígonos en el frame"""
        result = frame.copy()
        
        # Dibujar polígonos completados
        for i, (lane_name, points) in enumerate(self.completed_polygons.items()):
            if points and len(points) >= 3:
                pts = np.array(points, np.int32)
                color = self.lane_colors[i % len(self.lane_colors)]
                
                # Llenar polígono con transparencia
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
                
                # Contorno del polígono
                cv2.polylines(result, [pts], True, color, 2)
                
                # Etiqueta del carril
                if points:
                    center_x = sum(p[0] for p in points) // len(points)
                    center_y = sum(p[1] for p in points) // len(points)
                    cv2.putText(result, lane_name, (center_x - 10, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Dibujar puntos del polígono actual
        for i, point in enumerate(self.current_points):
            color = self.lane_colors[self.current_lane % len(self.lane_colors)]
            cv2.circle(result, point, 5, color, -1)
            cv2.putText(result, str(i + 1), (point[0] + 10, point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Líneas entre puntos actuales
        if len(self.current_points) > 1:
            color = self.lane_colors[self.current_lane % len(self.lane_colors)]
            for i in range(len(self.current_points) - 1):
                cv2.line(result, self.current_points[i], self.current_points[i + 1], color, 2)
        
        # Información en pantalla
        info_y = 30
        if self.current_lane < len(self.key_names):
            current_lane_name = self.key_names[self.current_lane]
            cv2.putText(result, f"🎯 Calibrando: {current_lane_name} ({self.current_lane + 1}/{len(self.key_names)})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            
            cv2.putText(result, f"Puntos: {len(self.current_points)}/4", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv2.putText(result, "✅ ¡Calibración completa! 's' = Guardar, 'r' = Reiniciar", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instrucciones
        instructions = [
            "Click izquierdo: Agregar punto",
            "Click derecho: Eliminar último punto", 
            "'s': Guardar polígonos",
            "'r': Reiniciar carril actual",
            "'q': Salir sin guardar"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(result, instruction, (10, result.shape[0] - 20 - (len(instructions) - i - 1) * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result
    
    def save_polygons(self):
        """Guardar polígonos en configuración"""
        try:
            # Verificar que todos los polígonos estén completos
            incomplete = []
            for lane_name in self.key_names:
                if lane_name not in self.completed_polygons or len(self.completed_polygons[lane_name]) < 3:
                    incomplete.append(lane_name)
            
            if incomplete:
                print(f"❌ Polígonos incompletos: {', '.join(incomplete)}")
                return False
            
            # Guardar cada polígono
            for lane_name, points in self.completed_polygons.items():
                section_name = f'LANE_POLYGON_{lane_name}'
                
                # Limpiar sección existente
                if self.config_manager.config.has_section(section_name):
                    self.config_manager.config.remove_section(section_name)
                
                self.config_manager.config.add_section(section_name)
                
                # Guardar puntos
                for i, (x, y) in enumerate(points):
                    self.config_manager.config.set(section_name, f'point_{i}_x', str(x))
                    self.config_manager.config.set(section_name, f'point_{i}_y', str(y))
                
                self.config_manager.config.set(section_name, 'point_count', str(len(points)))
            
            # Escribir archivo
            self.config_manager.save_config()
            
            print("✅ Polígonos guardados exitosamente")
            print("\n📊 NUEVOS POLÍGONOS:")
            self._show_final_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando polígonos: {e}")
            return False
    
    def _show_final_stats(self):
        """Mostrar estadísticas finales"""
        total_area_old = 0
        total_area_new = 0
        
        print("COMPARACIÓN ANTES/DESPUÉS:")
        for lane_name in self.key_names:
            # Área antigua
            if lane_name in self.lane_polygons and self.lane_polygons[lane_name]:
                old_pts = np.array(self.lane_polygons[lane_name], np.int32)
                old_area = cv2.contourArea(old_pts)
                total_area_old += old_area
            else:
                old_area = 0
            
            # Área nueva
            if lane_name in self.completed_polygons:
                new_pts = np.array(self.completed_polygons[lane_name], np.int32)
                new_area = cv2.contourArea(new_pts)
                total_area_new += new_area
            else:
                new_area = 0
            
            if old_area > 0:
                reduction = ((old_area - new_area) / old_area) * 100
                print(f"  🔹 {lane_name}: {old_area:.0f}px² → {new_area:.0f}px² ({reduction:+.1f}%)")
            else:
                print(f"  🔹 {lane_name}: NUEVO → {new_area:.0f}px²")
        
        if total_area_old > 0:
            total_reduction = ((total_area_old - total_area_new) / total_area_old) * 100
            print(f"\n📊 REDUCCIÓN TOTAL: {total_reduction:+.1f}% ({total_area_old:.0f} → {total_area_new:.0f}px²)")
            
            if total_reduction > 0:
                print(f"🚀 Estimación de mejora de rendimiento: ~{total_reduction * 0.8:.1f}%")
            else:
                print("⚠️ Los polígonos son más grandes - puede ser más lento")
    
    def run_calibration(self):
        """Ejecutar calibración interactiva"""
        cv2.namedWindow('Polygon Calibrator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Polygon Calibrator', 1400, 800)
        cv2.setMouseCallback('Polygon Calibrator', self.mouse_callback)
        
        print(f"\n🎯 INICIANDO CALIBRACIÓN")
        print(f"Carril actual: {self.key_names[0]} (1/{len(self.key_names)})")
        print("Define 4 puntos por carril haciendo click en las esquinas")
        
        while True:
            frame = self.screen_capture.capture_frame()
            if frame is None:
                continue
            
            result = self.draw_polygons(frame)
            cv2.imshow('Polygon Calibrator', result)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("❌ Calibración cancelada")
                break
            elif key == ord('s'):
                if self.save_polygons():
                    print("✅ ¡Polígonos guardados! Prueba el detector ahora")
                    break
            elif key == ord('r'):
                # Reiniciar carril actual
                self.current_points = []
                if self.current_lane < len(self.key_names):
                    lane_name = self.key_names[self.current_lane]
                    print(f"🔄 Reiniciando carril {lane_name}")
        
        cv2.destroyAllWindows()

def main():
    calibrator = PolygonCalibrator()
    calibrator.run_calibration()

if __name__ == "__main__":
    main() 