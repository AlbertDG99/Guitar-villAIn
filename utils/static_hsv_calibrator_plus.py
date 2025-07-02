#!/usr/bin/env python3
"""
🎨 Calibrador HSV Plus - Guitar Hero IA
Detección avanzada de cajas con ventana de control independiente
"""

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import Scale, Label, Frame, Button, IntVar
import threading
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class ControlWindow:
    """Ventana de control con parámetros HSV y morfología"""
    
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.root = tk.Tk()
        self.root.title("🎛️ Control HSV Plus")
        self.root.geometry("350x600")
        self.setup_variables()
        self.create_widgets()
    
    def setup_variables(self):
        """Variables tkinter con valores optimizados"""
        # Verde - el problema principal
        self.g_h_min = IntVar(value=49)
        self.g_h_max = IntVar(value=56) 
        self.g_s_min = IntVar(value=49)
        self.g_s_max = IntVar(value=216)
        self.g_v_min = IntVar(value=120)  # REDUCIDO de 229 a 120!
        self.g_v_max = IntVar(value=255)
        
        # Amarillo - restrictivo como tienes
        self.y_h_min = IntVar(value=28)
        self.y_h_max = IntVar(value=29)
        self.y_v_min = IntVar(value=231)
        self.y_v_max = IntVar(value=238)
        
        # Morfología para unir fragmentos
        self.close_size = IntVar(value=8)   # Rellenar huecos grandes
        self.dilate_size = IntVar(value=6)  # Expandir
        self.min_area = IntVar(value=80)
        self.max_area = IntVar(value=4000)
    
    def create_widgets(self):
        """Interfaz de control"""
        # VERDE
        Label(self.root, text="🟢 VERDE", font=('Arial', 14, 'bold'), fg='green').pack(pady=5)
        
        Scale(self.root, label="H Min", from_=0, to=179, orient='horizontal',
              variable=self.g_h_min, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="H Max", from_=0, to=179, orient='horizontal',
              variable=self.g_h_max, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="S Min", from_=0, to=255, orient='horizontal',
              variable=self.g_s_min, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="S Max", from_=0, to=255, orient='horizontal',
              variable=self.g_s_max, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="V Min ⭐", from_=0, to=255, orient='horizontal',
              variable=self.g_v_min, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="V Max", from_=0, to=255, orient='horizontal',
              variable=self.g_v_max, command=self.on_change).pack(fill='x', padx=10)
        
        # AMARILLO
        Label(self.root, text="🟡 AMARILLO", font=('Arial', 14, 'bold'), fg='orange').pack(pady=(20,5))
        
        Scale(self.root, label="Y H Min", from_=0, to=179, orient='horizontal',
              variable=self.y_h_min, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="Y H Max", from_=0, to=179, orient='horizontal',
              variable=self.y_h_max, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="Y V Min", from_=0, to=255, orient='horizontal',
              variable=self.y_v_min, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="Y V Max", from_=0, to=255, orient='horizontal',
              variable=self.y_v_max, command=self.on_change).pack(fill='x', padx=10)
        
        # MORFOLOGÍA
        Label(self.root, text="🔧 MORFOLOGÍA", font=('Arial', 14, 'bold')).pack(pady=(20,5))
        
        Scale(self.root, label="🔗 Close (rellenar)", from_=1, to=20, orient='horizontal',
              variable=self.close_size, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="➕ Dilate (expandir)", from_=1, to=15, orient='horizontal',
              variable=self.dilate_size, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="🏠 Área Min", from_=10, to=500, orient='horizontal',
              variable=self.min_area, command=self.on_change).pack(fill='x', padx=10)
        Scale(self.root, label="🏠 Área Max", from_=1000, to=10000, orient='horizontal',
              variable=self.max_area, command=self.on_change).pack(fill='x', padx=10)
        
        # BOTONES
        button_frame = Frame(self.root)
        button_frame.pack(fill='x', pady=20, padx=10)
        
        Button(button_frame, text="💾 Guardar", command=self.save, 
               bg='green', fg='white', font=('Arial', 11, 'bold')).pack(side='left', padx=5)
        Button(button_frame, text="🔄 Reset", command=self.reset, 
               bg='orange', fg='white', font=('Arial', 11, 'bold')).pack(side='left', padx=5)
        Button(button_frame, text="❌ Cerrar", command=self.close, 
               bg='red', fg='white', font=('Arial', 11, 'bold')).pack(side='right', padx=5)
    
    def on_change(self, value=None):
        """Actualizar parámetros en tiempo real"""
        params = {
            'green_lower': [self.g_h_min.get(), self.g_s_min.get(), self.g_v_min.get()],
            'green_upper': [self.g_h_max.get(), self.g_s_max.get(), self.g_v_max.get()],
            'yellow_lower': [self.y_h_min.get(), 100, self.y_v_min.get()],
            'yellow_upper': [self.y_h_max.get(), 255, self.y_v_max.get()],
            'close_size': self.close_size.get(),
            'dilate_size': self.dilate_size.get(),
            'min_area': self.min_area.get(),
            'max_area': self.max_area.get()
        }
        self.calibrator.update_params(params)
    
    def save(self):
        """Guardar configuración"""
        self.calibrator.save_config()
        print("💾 Configuración guardada")
    
    def reset(self):
        """Reset valores"""
        self.g_v_min.set(120)
        self.close_size.set(8)
        self.dilate_size.set(6)
        self.on_change()
        print("🔄 Reset aplicado")
    
    def close(self):
        """Cerrar ventana"""
        self.calibrator.control_active = False
        self.root.destroy()
    
    def run(self):
        """Ejecutar ventana"""
        self.root.mainloop()


class StaticHSVCalibratorPlus:
    """Calibrador HSV avanzado para estrellas fragmentadas"""
    
    def __init__(self):
        # Cargar imagen
        self.image_path = Path(__file__).parent.parent / 'data/templates/image.png'
        if not self.image_path.exists():
            print(f"❌ No se encontró {self.image_path}")
            sys.exit(1)
        
        self.original = cv2.imread(str(self.image_path))
        if self.original is None:
            print("❌ Error cargando imagen")
            sys.exit(1)
            
        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        print(f"✅ Imagen: {self.original.shape[1]}x{self.original.shape[0]}")
        
        # Parámetros optimizados para tus estrellas fragmentadas
        self.params = {
            'green_lower': [49, 49, 120],  # V_min REDUCIDO!
            'green_upper': [56, 216, 255],
            'yellow_lower': [28, 100, 231],
            'yellow_upper': [29, 255, 238],
            'close_size': 8,   # Agresivo para rellenar
            'dilate_size': 6,  # Expandir para conectar
            'min_area': 80,
            'max_area': 4000
        }
        
        self.control_active = False
        self.info_text = ""
        self.view_mode = 0  # 0=cajas, 1=verde, 2=amarilla
        
        print("🎨 Calibrador HSV Plus listo")
        print("🎯 Optimizado para estrellas fragmentadas")
    
    def update_params(self, new_params):
        """Actualizar parámetros desde control"""
        self.params.update(new_params)
    
    def detect_color_boxes(self, color='green'):
        """Detectar cajas de un color específico - GLOBAL"""
        if color == 'green':
            lower = np.array(self.params['green_lower'])
            upper = np.array(self.params['green_upper'])
            box_color = (0, 255, 0)
        else:
            lower = np.array(self.params['yellow_lower'])
            upper = np.array(self.params['yellow_upper'])
            box_color = (0, 255, 255)
        
        # Máscara HSV
        mask = cv2.inRange(self.hsv, lower, upper)
        
        # MORFOLOGÍA AGRESIVA para estrellas fragmentadas
        # 1. CLOSE - rellenar huecos entre fragmentos
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (self.params['close_size'], self.params['close_size']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 2. DILATE - expandir para conectar fragmentos cercanos
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (self.params['dilate_size'], self.params['dilate_size']))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # 3. OPEN pequeño - limpiar ruido sin afectar formas grandes
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        
        # Detectar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['min_area'] <= area <= self.params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                # Filtro de forma básico
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio <= 5.0:  # No muy alargado
                    boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 
                                'area': area, 'color': box_color})
        
        return boxes, mask
    
    def process_frame(self):
        """Procesar frame con detección"""
        # Detectar ambos colores
        green_boxes, green_mask = self.detect_color_boxes('green')
        yellow_boxes, yellow_mask = self.detect_color_boxes('yellow')
        
        # Imagen de salida según modo
        if self.view_mode == 0:    # Original + cajas
            output = self.original.copy()
            show_boxes = True
        elif self.view_mode == 1:  # Máscara verde
            output = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            show_boxes = False
        elif self.view_mode == 2:  # Máscara amarilla
            output = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            show_boxes = False
        
        # Dibujar cajas
        if show_boxes:
            for box in green_boxes:
                cv2.rectangle(output, (box['x'], box['y']), 
                            (box['x'] + box['w'], box['y'] + box['h']), box['color'], 2)
                cv2.putText(output, f"G:{int(box['area'])}", (box['x'], box['y']-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, box['color'], 1)
            
            for box in yellow_boxes:
                cv2.rectangle(output, (box['x'], box['y']), 
                            (box['x'] + box['w'], box['y'] + box['h']), box['color'], 2)
                cv2.putText(output, f"Y:{int(box['area'])}", (box['x'], box['y']-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, box['color'], 1)
        
        # Info overlay
        self.add_overlay(output, len(green_boxes), len(yellow_boxes))
        return output
    
    def add_overlay(self, img, green_count, yellow_count):
        """Info en pantalla"""
        # Fondo
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (550, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Info principal
        cv2.putText(img, f"🟢 VERDES: {green_count} | 🟡 AMARILLAS: {yellow_count}", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Parámetros actuales
        g_v_min = self.params['green_lower'][2]
        cv2.putText(img, f"Verde V_min: {g_v_min} | Close: {self.params['close_size']} | Dilate: {self.params['dilate_size']}", 
                   (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Modo vista
        modes = ["📦 CAJAS", "🟢 VERDE", "🟡 AMARILLA"]
        cv2.putText(img, f"Vista: {modes[self.view_mode]}", 
                   (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controles
        cv2.putText(img, "CTRL: 'c'=Control | 's'=Guardar | '1'/'2'/'3'=Vista | 'q'=Salir", 
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Info del píxel clickeado"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < self.hsv.shape[0] and 0 <= x < self.hsv.shape[1]:
                hsv_pixel = self.hsv[y, x]
                bgr_pixel = self.original[y, x]
                print(f"🎯 Píxel ({x},{y}) - HSV: {hsv_pixel} | BGR: {bgr_pixel}")
    
    def open_control(self):
        """Abrir ventana de control"""
        if not self.control_active:
            self.control_active = True
            def run_control():
                control = ControlWindow(self)
                control.run()
                self.control_active = False
            threading.Thread(target=run_control, daemon=True).start()
            print("🎛️ Ventana de control abierta")
    
    def save_config(self):
        """Guardar configuración a archivo"""
        file_path = Path(__file__).parent.parent / 'hsv_ranges_optimized_plus.txt'
        try:
            with open(file_path, 'w') as f:
                f.write("# HSV Optimizado Plus - Estrellas Fragmentadas\n\n")
                gl = self.params['green_lower']
                gu = self.params['green_upper']
                yl = self.params['yellow_lower']
                yu = self.params['yellow_upper']
                
                f.write(f"green_lower = np.array([{gl[0]}, {gl[1]}, {gl[2]}])\n")
                f.write(f"green_upper = np.array([{gu[0]}, {gu[1]}, {gu[2]}])\n")
                f.write(f"yellow_lower = np.array([{yl[0]}, {yl[1]}, {yl[2]}])\n")
                f.write(f"yellow_upper = np.array([{yu[0]}, {yu[1]}, {yu[2]}])\n\n")
                f.write(f"# Morfología optimizada\n")
                f.write(f"close_size = {self.params['close_size']}\n")
                f.write(f"dilate_size = {self.params['dilate_size']}\n")
                f.write(f"min_area = {self.params['min_area']}\n")
                f.write(f"max_area = {self.params['max_area']}\n")
            print(f"💾 Guardado en: {file_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run(self):
        """Ejecutar calibrador"""
        print("\n" + "="*60)
        print("🎨 HSV CALIBRATOR PLUS")
        print("="*60)
        print("🎯 OBJETIVO: Detectar esas 2 estrellas verdes fragmentadas")
        print("💡 Configuración inicial: V_min verde = 120 (era 229)")
        print("🔧 Morfología agresiva: Close=8, Dilate=6")
        print("")
        print("CONTROLES:")
        print("- 'c': Abrir ventana de control")
        print("- '1': Vista Original + Cajas")
        print("- '2': Vista Máscara Verde")
        print("- '3': Vista Máscara Amarilla")
        print("- 's': Guardar configuración")
        print("- 'q': Salir")
        print("- Click: Info del píxel")
        print("="*60)
        
        cv2.namedWindow('HSV Calibrator Plus')
        cv2.setMouseCallback('HSV Calibrator Plus', self.mouse_callback)
        
        try:
            while True:
                output = self.process_frame()
                cv2.imshow('HSV Calibrator Plus', output)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.open_control()
                elif key == ord('s'):
                    self.save_config()
                elif key == ord('1'):
                    self.view_mode = 0
                    print("📦 Vista: Original + Cajas")
                elif key == ord('2'):
                    self.view_mode = 1
                    print("🟢 Vista: Máscara Verde")
                elif key == ord('3'):
                    self.view_mode = 2
                    print("🟡 Vista: Máscara Amarilla")
                    
        except KeyboardInterrupt:
            print("\n⛔ Interrumpido")
        finally:
            cv2.destroyAllWindows()
            print("👋 Calibrador cerrado")


def main():
    """Función principal"""
    try:
        calibrator = StaticHSVCalibratorPlus()
        calibrator.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
