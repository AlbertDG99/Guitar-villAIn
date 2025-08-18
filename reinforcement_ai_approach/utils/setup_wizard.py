#!/usr/bin/env python3
"""
Interactive setup wizard for RL approach:
- Select capture region (draw rectangle)
- Define lane polygons (S,D,F,J,K,L) by clicking points and pressing ENTER to confirm each lane
- Define SCORE and COMBO ROIs (draw rectangles)

Saves values into config.ini so they persist between runs.
"""

import cv2
import numpy as np
from mss import mss
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os
import configparser

# Make absolute imports work even if executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # reinforcement_ai_approach
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reinforcement_ai_approach.src.utils.config_manager import ConfigManager


def get_screen_dimensions() -> Tuple[int, int]:
    """Obtiene las dimensiones de la pantalla principal."""
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        # Fallback a tamaños estándar
        return 1920, 1080


def select_game_window() -> dict:
    """Permite al usuario seleccionar la región del juego dibujando un rectángulo."""
    print("\n🎮 Selección de Región del Juego")
    print("=" * 50)
    print("💡 Dibuja un rectángulo alrededor de la ventana del juego")
    print("   Esta será la zona que se capturará para detectar notas")
    
    with mss() as sct:
        monitor = sct.monitors[1]  # Monitor principal
        full_region = {
            'left': monitor['left'],
            'top': monitor['top'],
            'width': monitor['width'],
            'height': monitor['height']
        }
        
        # Selección en vivo del área de la pantalla
        window = 'Selección de Región del Juego'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("📱 Dibuja un rectángulo alrededor de la ventana del juego (en vivo)")
        print("   Presiona 's' para guardar, 'q' para cancelar")
        rect = draw_rect_live(window, sct, full_region)
        cv2.destroyAllWindows()
        
        if rect:
            x, y, w, h = rect
            return {
                'left': x,
                'top': y,
                'width': w,
                'height': h
            }
        else:
            print("⚠️ No se seleccionó región, usando monitor completo")
            return full_region





def draw_rect_live(window: str, sct: mss, region: dict) -> Optional[Tuple[int, int, int, int]]:
    """Selección de rectángulo con vista previa EN VIVO y ajuste proporcional.
    Devuelve coordenadas en el espacio ORIGINAL de la región capturada (x, y, w, h).
    """
    state = {
        "active": False,
        "ix": 0,
        "iy": 0,
        "rect": None,  # (x, y, w, h) en coords originales
        "scale": 1.0,
        "offset_x": 0,
        "offset_y": 0,
        "disp_w": 0,
        "disp_h": 0,
    }

    def to_original_coords(px: int, py: int) -> Tuple[int, int]:
        x = int((px - state["offset_x"]) / max(state["scale"], 1e-6))
        y = int((py - state["offset_y"]) / max(state["scale"], 1e-6))
        # Clamp
        x = max(0, min(region['width'] - 1, x))
        y = max(0, min(region['height'] - 1, y))
        return x, y

    def to_display_coords(ox: int, oy: int) -> Tuple[int, int]:
        dx = int(ox * state["scale"]) + state["offset_x"]
        dy = int(oy * state["scale"]) + state["offset_y"]
        return dx, dy

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["active"] = True
            ox, oy = to_original_coords(x, y)
            state["ix"], state["iy"] = ox, oy
            state["rect"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["active"]:
            ox, oy = to_original_coords(x, y)
            x0, y0 = state["ix"], state["iy"]
            state["rect"] = (
                min(x0, ox),
                min(y0, oy),
                abs(ox - x0),
                abs(oy - y0),
            )
        elif event == cv2.EVENT_LBUTTONUP:
            state["active"] = False
            ox, oy = to_original_coords(x, y)
            x0, y0 = state["ix"], state["iy"]
            state["rect"] = (
                min(x0, ox),
                min(y0, oy),
                abs(ox - x0),
                abs(oy - y0),
            )

    cv2.setMouseCallback(window, _cb)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    init_w, init_h = get_screen_dimensions()
    cv2.resizeWindow(window, int(init_w * 0.8), int(init_h * 0.8))

    while True:
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fh, fw = frame.shape[:2]

        # Obtener tamaño actual de la ventana (área de imagen)
        _, _, ww, wh = cv2.getWindowImageRect(window)
        if ww <= 0 or wh <= 0:
            ww, wh = int(init_w * 0.8), int(init_h * 0.8)

        scale = min(ww / fw, wh / fh)
        new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        off_x = (ww - new_w) // 2
        off_y = (wh - new_h) // 2

        state["scale"] = scale
        state["offset_x"] = off_x
        state["offset_y"] = off_y
        state["disp_w"] = ww
        state["disp_h"] = wh

        canvas = np.zeros((max(1, wh), max(1, ww), 3), dtype=np.uint8)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        # Dibujar rectángulo si existe
        if state["rect"] is not None:
            rx, ry, rw, rh = state["rect"]
            p1 = to_display_coords(rx, ry)
            p2 = to_display_coords(rx + rw, ry + rh)
            cv2.rectangle(canvas, p1, p2, (0, 255, 0), 2)

        cv2.putText(canvas, "Arrastra para seleccionar. 's' guardar, 'q' cancelar",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('s') and state["rect"]:
            return state["rect"]


def draw_polygon_live(window: str, sct: mss, region: dict, label: str) -> Optional[List[Tuple[int, int]]]:
    """Selección de POLÍGONO con vista previa EN VIVO y ajuste proporcional.
    Devuelve lista de puntos en coordenadas ORIGINALES de la región capturada.
    """
    points: List[Tuple[int, int]] = []  # originales
    state = {
        "scale": 1.0,
        "offset_x": 0,
        "offset_y": 0,
        "disp_w": 0,
        "disp_h": 0,
    }

    def to_original_coords(px: int, py: int) -> Tuple[int, int]:
        x = int((px - state["offset_x"]) / max(state["scale"], 1e-6))
        y = int((py - state["offset_y"]) / max(state["scale"], 1e-6))
        x = max(0, min(region['width'] - 1, x))
        y = max(0, min(region['height'] - 1, y))
        return x, y

    def to_display_coords(ox: int, oy: int) -> Tuple[int, int]:
        dx = int(ox * state["scale"]) + state["offset_x"]
        dy = int(oy * state["scale"]) + state["offset_y"]
        return dx, dy

    def _cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = to_original_coords(x, y)
            points.append((ox, oy))

    cv2.setMouseCallback(window, _cb)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    init_w, init_h = get_screen_dimensions()
    cv2.resizeWindow(window, int(init_w * 0.8), int(init_h * 0.8))

    while True:
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fh, fw = frame.shape[:2]

        # Tamaño actual de ventana
        _, _, ww, wh = cv2.getWindowImageRect(window)
        if ww <= 0 or wh <= 0:
            ww, wh = int(init_w * 0.8), int(init_h * 0.8)

        scale = min(ww / fw, wh / fh)
        new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        off_x = (ww - new_w) // 2
        off_y = (wh - new_h) // 2
        state["scale"], state["offset_x"], state["offset_y"] = scale, off_x, off_y
        state["disp_w"], state["disp_h"] = ww, wh

        canvas = np.zeros((max(1, wh), max(1, ww), 3), dtype=np.uint8)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        # Dibujar puntos y polilíneas
        if points:
            disp_points = [to_display_coords(px, py) for (px, py) in points]
            for p in disp_points:
                cv2.circle(canvas, p, 3, (0, 255, 0), -1)
            if len(disp_points) > 1:
                for i in range(len(disp_points) - 1):
                    cv2.line(canvas, disp_points[i], disp_points[i + 1], (0, 255, 0), 1)
            if len(disp_points) >= 3:
                cv2.line(canvas, disp_points[-1], disp_points[0], (0, 255, 0), 1)
                cv2.polylines(canvas, [np.array(disp_points, np.int32)], True, (0, 255, 0), 2)

        cv2.putText(canvas, f"Lane {label}: clic para puntos, ENTER confirmar, 'r' reset, 'u' undo, 'q' salir",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Points: {len(points)} (min 3)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('r'):
            points.clear()
        if key == ord('u') and points:
            points.pop()
        if key == 13 and len(points) >= 3:  # ENTER
            return points


def main():
    cfg_path = Path(__file__).parent.parent / 'config' / 'config.ini'
    config = ConfigManager(config_path=str(cfg_path))

    # Paso 1: Seleccionar la región del juego
    window_region = select_game_window()
    
    print(f"\n📱 Región del juego seleccionada:")
    print(f"   Posición: ({window_region['left']}, {window_region['top']})")
    print(f"   Tamaño: {window_region['width']} x {window_region['height']}")
    
    print("💡 Asegúrate de que la ventana del juego esté visible y activa")
    input("Presiona ENTER cuando estés listo para continuar...")
    
    with mss() as sct:
        # Usar la región de la ventana del juego seleccionada
        region = window_region

        window = 'Setup Wizard'
        cv2.namedWindow(window)

        # La región seleccionada ya es la zona de captura
        print("\n🎯 La región seleccionada será la zona de captura para detectar notas")
        print(f"   Región: ({region['left']}, {region['top']}) - {region['width']}x{region['height']}")
        
        # Guardar directamente la región seleccionada como zona de captura
        if not config.config.has_section('CAPTURE'):
            config.config.add_section('CAPTURE')
        config.config.set('CAPTURE', 'game_left', str(region['left']))
        config.config.set('CAPTURE', 'game_top', str(region['top']))
        config.config.set('CAPTURE', 'game_width', str(region['width']))
        config.config.set('CAPTURE', 'game_height', str(region['height']))
        config.config.set('CAPTURE', 'last_confirmed_width', str(region['width']))
        config.config.set('CAPTURE', 'last_confirmed_height', str(region['height']))
        
        print("✅ Región de captura configurada automáticamente")

        # Paso 1: lane polygons (dentro de la región de captura)
        print("\n🎵 Paso 1: Configurar polígonos de carriles (S, D, F, J, K, L)")
        print("   (Dibuja los polígonos donde aparecen las notas en cada carril)")
        
        lane_order = ['S', 'D', 'F', 'J', 'K', 'L']
        for lane in lane_order:
            print(f"\n🎯 Configurando carril {lane}...")
            poly = draw_polygon_live(window, sct, region, lane)
            if poly is None:
                print(f"⚠️ Carril {lane} saltado")
                continue
            section = f'LANE_POLYGON_{lane}'
            if not config.config.has_section(section):
                config.config.add_section(section)
            config.config.set(section, 'point_count', str(len(poly)))
            for i, (px, py) in enumerate(poly):
                config.config.set(section, f'point_{i}_x', str(px))
                config.config.set(section, f'point_{i}_y', str(py))
            print(f"✅ Carril {lane} configurado con {len(poly)} puntos")

        # Paso 2: SCORE ROI
        print("\n🏆 Paso 2: Configurar región de puntuación (SCORE)")
        score_rect = draw_rect_live(window, sct, region)
        if score_rect:
            x, y, w, h = score_rect
            if not config.config.has_section('SCORE'):
                config.config.add_section('SCORE')
            config.config.set('SCORE', 'score_region_relative', str({'left': x, 'top': y, 'width': w, 'height': h}))
            # Opcional: también almacenamos absoluta por conveniencia
            config.config.set('SCORE', 'score_region', str({'left': region['left'] + x, 'top': region['top'] + y, 'width': w, 'height': h}))
            print(f"✅ Región de SCORE configurada: ({x}, {y}) - {w}x{h}")

        # Paso 3: COMBO ROI
        print("\n🔥 Paso 3: Configurar región de combo (COMBO)")
        combo_rect = draw_rect_live(window, sct, region)
        if combo_rect:
            x, y, w, h = combo_rect
            if not config.config.has_section('COMBO'):
                config.config.add_section('COMBO')
            config.config.set('COMBO', 'combo_region', str({'left': x, 'top': y, 'width': w, 'height': h}))
            print(f"✅ Región de COMBO configurada: ({x}, {y}) - {w}x{h}")

        # Paso 4: SONG TIME ROI (top-right timer)
        print("\n⏰ Paso 4: Configurar región de tiempo de canción (SONG TIME)")
        time_rect = draw_rect_live(window, sct, region)
        if time_rect:
            x, y, w, h = time_rect
            if not config.config.has_section('SONG'):
                config.config.add_section('SONG')
            config.config.set('SONG', 'song_time_region', str({'left': x, 'top': y, 'width': w, 'height': h}))
            print(f"✅ Región de TIEMPO configurada: ({x}, {y}) - {w}x{h}")

        # Guardar configuración (RL approach)
        config.save_config()

        # Replicar secciones relevantes al config del Color Pattern Approach
        try:
            color_cfg_path = REPO_ROOT / 'color_pattern_approach' / 'config.ini'
            cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(';', '#'))
            if color_cfg_path.exists():
                cp.read(color_cfg_path, encoding='utf-8')

            # Copiar secciones relevantes
            def copy_section(sec: str):
                if config.config.has_section(sec):
                    if cp.has_section(sec):
                        cp.remove_section(sec)
                    cp.add_section(sec)
                    for k, v in config.config.items(sec):
                        cp.set(sec, k, v)

            copy_section('CAPTURE')
            copy_section('SCORE')
            copy_section('COMBO')
            copy_section('SONG')
            for lane in ['S','D','F','J','K','L']:
                copy_section(f'LANE_POLYGON_{lane}')

            with open(color_cfg_path, 'w', encoding='utf-8') as f:
                cp.write(f)
            print("📝 Config sincronizado también en color_pattern_approach/config.ini")
        except Exception as e:
            print(f"⚠️ No se pudo sincronizar config al color_pattern_approach: {e}")

        print("\n🎉 ¡Configuración completada exitosamente!")
        print("✅ Todos los polígonos y regiones han sido guardados y sincronizados")
        print("🚀 Ya puedes ejecutar el bot de Guitar Hero")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


