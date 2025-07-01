#!/usr/bin/env python3
"""
🎨 Calibrador HSV Estático - Guitar Hero IA
Calibra rangos de color HSV usando screenshot estático sin pausar el juego
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Añadir src al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from utils.config_manager import ConfigManager

class StaticHSVCalibrator:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.image_path = Path(__file__).parent.parent / 'data/templates/image.png'
        
        # Verificar que existe la imagen
        if not self.image_path.exists():
            print(f"❌ ERROR: No se encontró la imagen en {self.image_path}")
            print("💡 SUGERENCIA: Toma un screenshot del juego y guárdalo como 'image.png' en data/templates/")
            sys.exit(1)
            
        # Cargar imagen
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            print(f"❌ ERROR: No se pudo cargar la imagen {self.image_path}")
            sys.exit(1)
            
        print(f"✅ Imagen cargada: {self.image_path}")
        print(f"📐 Dimensiones: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        
        # Convertir a HSV
        self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Rangos HSV iniciales
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])
        self.green_lower = np.array([25, 40, 40])
        self.green_upper = np.array([95, 255, 255])
        
        # Cargar rangos optimizados si existen
        self.load_optimized_ranges()
        
        # Variables para el mouse callback
        self.selected_color = None
        self.info_text = ""
        
        print("🎨 Calibrador HSV Estático iniciado")
        print("✨ Usa los sliders para ajustar los rangos de color")

    def load_optimized_ranges(self):
        """Cargar rangos HSV optimizados si existen"""
        hsv_file = Path(__file__).parent.parent / 'hsv_ranges_optimized.txt'
        if hsv_file.exists():
            try:
                with open(hsv_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'YELLOW' in line and 'lower' in line:
                            values = line.split('[')[1].split(']')[0].split()
                            self.yellow_lower = np.array([int(v) for v in values])
                        elif 'YELLOW' in line and 'upper' in line:
                            values = line.split('[')[1].split(']')[0].split()
                            self.yellow_upper = np.array([int(v) for v in values])
                        elif 'GREEN' in line and 'lower' in line:
                            values = line.split('[')[1].split(']')[0].split()
                            self.green_lower = np.array([int(v) for v in values])
                        elif 'GREEN' in line and 'upper' in line:
                            values = line.split('[')[1].split(']')[0].split()
                            self.green_upper = np.array([int(v) for v in values])
                print("✅ Rangos HSV optimizados cargados")
            except Exception as e:
                print(f"⚠️ Error cargando rangos optimizados: {e}")

    def save_optimized_ranges(self):
        """Guardar rangos HSV calibrados"""
        hsv_file = Path(__file__).parent.parent / 'hsv_ranges_optimized.txt'
        try:
            with open(hsv_file, 'w') as f:
                f.write("# Rangos HSV Optimizados - Guitar Hero IA\n")
                f.write("# Generado por Static HSV Calibrator\n\n")
                f.write(f"YELLOW lower: {self.yellow_lower}\n")
                f.write(f"YELLOW upper: {self.yellow_upper}\n")
                f.write(f"GREEN lower: {self.green_lower}\n")
                f.write(f"GREEN upper: {self.green_upper}\n")
            print(f"💾 Rangos guardados en: {hsv_file}")
            return True
        except Exception as e:
            print(f"❌ Error guardando rangos: {e}")
            return False

    def mouse_callback(self, event, x, y, flags, param):
        """Callback para mostrar información del píxel clickeado"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < self.hsv_image.shape[0] and 0 <= x < self.hsv_image.shape[1]:
                # Obtener valores HSV del píxel
                hsv_pixel = self.hsv_image[y, x]
                bgr_pixel = self.original_image[y, x]
                
                self.selected_color = hsv_pixel
                self.info_text = f"Píxel ({x},{y}) - HSV: {hsv_pixel} | BGR: {bgr_pixel}"
                print(f"🎯 {self.info_text}")

    def create_trackbars(self):
        """Crear ventana con trackbars para calibración"""
        cv2.namedWindow('Calibrador HSV')
        cv2.setMouseCallback('Calibrador HSV', self.mouse_callback)
        
        # Trackbars para YELLOW
        cv2.createTrackbar('Y_H_Min', 'Calibrador HSV', int(self.yellow_lower[0]), 179, lambda x: None)
        cv2.createTrackbar('Y_S_Min', 'Calibrador HSV', int(self.yellow_lower[1]), 255, lambda x: None)
        cv2.createTrackbar('Y_V_Min', 'Calibrador HSV', int(self.yellow_lower[2]), 255, lambda x: None)
        cv2.createTrackbar('Y_H_Max', 'Calibrador HSV', int(self.yellow_upper[0]), 179, lambda x: None)
        cv2.createTrackbar('Y_S_Max', 'Calibrador HSV', int(self.yellow_upper[1]), 255, lambda x: None)
        cv2.createTrackbar('Y_V_Max', 'Calibrador HSV', int(self.yellow_upper[2]), 255, lambda x: None)
        
        # Trackbars para GREEN
        cv2.createTrackbar('G_H_Min', 'Calibrador HSV', int(self.green_lower[0]), 179, lambda x: None)
        cv2.createTrackbar('G_S_Min', 'Calibrador HSV', int(self.green_lower[1]), 255, lambda x: None)
        cv2.createTrackbar('G_V_Min', 'Calibrador HSV', int(self.green_lower[2]), 255, lambda x: None)
        cv2.createTrackbar('G_H_Max', 'Calibrador HSV', int(self.green_upper[0]), 179, lambda x: None)
        cv2.createTrackbar('G_S_Max', 'Calibrador HSV', int(self.green_upper[1]), 255, lambda x: None)
        cv2.createTrackbar('G_V_Max', 'Calibrador HSV', int(self.green_upper[2]), 255, lambda x: None)
        
        # Selector de modo
        cv2.createTrackbar('Modo', 'Calibrador HSV', 0, 2, lambda x: None)  # 0=Original, 1=Yellow, 2=Green

    def get_trackbar_values(self):
        """Obtener valores actuales de los trackbars"""
        # Yellow ranges
        y_h_min = cv2.getTrackbarPos('Y_H_Min', 'Calibrador HSV')
        y_s_min = cv2.getTrackbarPos('Y_S_Min', 'Calibrador HSV')
        y_v_min = cv2.getTrackbarPos('Y_V_Min', 'Calibrador HSV')
        y_h_max = cv2.getTrackbarPos('Y_H_Max', 'Calibrador HSV')
        y_s_max = cv2.getTrackbarPos('Y_S_Max', 'Calibrador HSV')
        y_v_max = cv2.getTrackbarPos('Y_V_Max', 'Calibrador HSV')
        
        # Green ranges
        g_h_min = cv2.getTrackbarPos('G_H_Min', 'Calibrador HSV')
        g_s_min = cv2.getTrackbarPos('G_S_Min', 'Calibrador HSV')
        g_v_min = cv2.getTrackbarPos('G_V_Min', 'Calibrador HSV')
        g_h_max = cv2.getTrackbarPos('G_H_Max', 'Calibrador HSV')
        g_s_max = cv2.getTrackbarPos('G_S_Max', 'Calibrador HSV')
        g_v_max = cv2.getTrackbarPos('G_V_Max', 'Calibrador HSV')
        
        # Modo de visualización
        mode = cv2.getTrackbarPos('Modo', 'Calibrador HSV')
        
        return {
            'yellow_lower': np.array([y_h_min, y_s_min, y_v_min]),
            'yellow_upper': np.array([y_h_max, y_s_max, y_v_max]),
            'green_lower': np.array([g_h_min, g_s_min, g_v_min]),
            'green_upper': np.array([g_h_max, g_s_max, g_v_max]),
            'mode': mode
        }

    def apply_hsv_filter(self, lower, upper):
        """Aplicar filtro HSV y retornar máscara"""
        mask = cv2.inRange(self.hsv_image, lower, upper)
        return mask

    def create_info_overlay(self, image, values):
        """Crear overlay con información"""
        overlay = image.copy()
        
        # Fondo semi-transparente para el texto
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Información de rangos
        y_offset = 30
        cv2.putText(image, f"🟡 YELLOW: [{values['yellow_lower']}] - [{values['yellow_upper']}]", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
        cv2.putText(image, f"🟢 GREEN:  [{values['green_lower']}] - [{values['green_upper']}]", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 35
        
        # Modo actual
        modes = ["ORIGINAL", "YELLOW MASK", "GREEN MASK"]
        cv2.putText(image, f"📺 MODO: {modes[values['mode']]}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 35
        
        # Información del píxel seleccionado
        if self.info_text:
            cv2.putText(image, f"🎯 {self.info_text}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Controles
        cv2.putText(image, "CONTROLES: Click=Info pixel | 's'=Guardar | 'r'=Reset | 'q'=Salir", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image

    def reset_ranges(self):
        """Resetear rangos a valores por defecto"""
        # Valores por defecto
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])
        self.green_lower = np.array([25, 40, 40])
        self.green_upper = np.array([95, 255, 255])
        
        # Actualizar trackbars
        cv2.setTrackbarPos('Y_H_Min', 'Calibrador HSV', int(self.yellow_lower[0]))
        cv2.setTrackbarPos('Y_S_Min', 'Calibrador HSV', int(self.yellow_lower[1]))
        cv2.setTrackbarPos('Y_V_Min', 'Calibrador HSV', int(self.yellow_lower[2]))
        cv2.setTrackbarPos('Y_H_Max', 'Calibrador HSV', int(self.yellow_upper[0]))
        cv2.setTrackbarPos('Y_S_Max', 'Calibrador HSV', int(self.yellow_upper[1]))
        cv2.setTrackbarPos('Y_V_Max', 'Calibrador HSV', int(self.yellow_upper[2]))
        
        cv2.setTrackbarPos('G_H_Min', 'Calibrador HSV', int(self.green_lower[0]))
        cv2.setTrackbarPos('G_S_Min', 'Calibrador HSV', int(self.green_lower[1]))
        cv2.setTrackbarPos('G_V_Min', 'Calibrador HSV', int(self.green_lower[2]))
        cv2.setTrackbarPos('G_H_Max', 'Calibrador HSV', int(self.green_upper[0]))
        cv2.setTrackbarPos('G_S_Max', 'Calibrador HSV', int(self.green_upper[1]))
        cv2.setTrackbarPos('G_V_Max', 'Calibrador HSV', int(self.green_upper[2]))
        
        print("🔄 Rangos HSV reseteados a valores por defecto")

    def run(self):
        """Ejecutar el calibrador"""
        self.create_trackbars()
        
        print("\n🎨 CALIBRADOR HSV ESTÁTICO INICIADO")
        print("=" * 50)
        print("📋 INSTRUCCIONES:")
        print("   • Usa los sliders para ajustar rangos HSV")
        print("   • Click en la imagen para ver valores HSV del píxel")
        print("   • Cambiar 'Modo' para ver diferentes vistas:")
        print("     - 0: Imagen original")
        print("     - 1: Máscara amarilla")
        print("     - 2: Máscara verde")
        print("   • 's': Guardar rangos calibrados")
        print("   • 'r': Reset rangos a por defecto")
        print("   • 'q': Salir")
        print("=" * 50)
        
        while True:
            # Obtener valores actuales
            values = self.get_trackbar_values()
            
            # Actualizar rangos internos
            self.yellow_lower = values['yellow_lower']
            self.yellow_upper = values['yellow_upper']
            self.green_lower = values['green_lower']
            self.green_upper = values['green_upper']
            
            # Crear imagen según el modo
            if values['mode'] == 0:  # Original
                display_image = self.original_image.copy()
            elif values['mode'] == 1:  # Yellow mask
                yellow_mask = self.apply_hsv_filter(values['yellow_lower'], values['yellow_upper'])
                display_image = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            elif values['mode'] == 2:  # Green mask
                green_mask = self.apply_hsv_filter(values['green_lower'], values['green_upper'])
                display_image = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            
            # Agregar overlay de información
            display_image = self.create_info_overlay(display_image, values)
            
            # Mostrar imagen
            cv2.imshow('Calibrador HSV', display_image)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.save_optimized_ranges():
                    print("✅ ¡Rangos HSV guardados exitosamente!")
                else:
                    print("❌ Error guardando rangos HSV")
            elif key == ord('r'):
                self.reset_ranges()
        
        cv2.destroyAllWindows()
        print("\n🎨 Calibrador HSV Estático finalizado")
        print("💾 Los rangos calibrados se han guardado para el sistema principal")

def main():
    """Función principal"""
    print("🎸 Guitar Hero IA - Calibrador HSV Estático")
    print("=" * 50)
    
    try:
        calibrator = StaticHSVCalibrator()
        calibrator.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Calibración interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 