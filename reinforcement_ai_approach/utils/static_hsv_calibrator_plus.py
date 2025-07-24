#!/usr/bin/env python3
"""
🎨 HSV Calibrator Plus - Guitar Hero AI
Advanced box detection with independent control window
"""

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import Scale, Label, Frame, Button, IntVar
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))


class ControlWindow:
    """Control window with HSV and morphology parameters"""

    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.root = tk.Tk()
        self.root.title("🎛️ Control HSV Plus")
        self.root.geometry("350x600")
        self.setup_variables()
        self.create_widgets()

    def setup_variables(self):
        """Tkinter variables with optimized values"""
        self.g_h_min = IntVar(value=49)
        self.g_h_max = IntVar(value=56)
        self.g_s_min = IntVar(value=49)
        self.g_s_max = IntVar(value=216)
        self.g_v_min = IntVar(value=120)
        self.g_v_max = IntVar(value=255)

        self.y_h_min = IntVar(value=28)
        self.y_h_max = IntVar(value=29)
        self.y_v_min = IntVar(value=231)
        self.y_v_max = IntVar(value=238)

        self.close_size = IntVar(value=8)
        self.dilate_size = IntVar(value=6)
        self.min_area = IntVar(value=80)
        self.max_area = IntVar(value=4000)

    def create_widgets(self):
        """Control interface"""
        Label(
            self.root,
            text="🟢 VERDE",
            font=(
                'Arial',
                14,
                'bold'),
            fg='green').pack(
            pady=5)

        Scale(
            self.root,
            label="H Min",
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.g_h_min,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="H Max",
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.g_h_max,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="S Min",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.g_s_min,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="S Max",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.g_s_max,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="V Min ⭐",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.g_v_min,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="V Max",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.g_v_max,
            command=self.on_change).pack(
            fill='x',
            padx=10)

        Label(
            self.root,
            text="🟡 AMARILLO",
            font=(
                'Arial',
                14,
                'bold'),
            fg='orange').pack(
            pady=(
                20,
                5))

        Scale(
            self.root,
            label="Y H Min",
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.y_h_min,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="Y H Max",
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.y_h_max,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="Y V Min",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.y_v_min,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="Y V Max",
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.y_v_max,
            command=self.on_change).pack(
            fill='x',
            padx=10)

        Label(
            self.root,
            text="🔧 MORPHOLOGY",
            font=(
                'Arial',
                14,
                'bold')).pack(
            pady=(
                20,
                5))

        Scale(
            self.root,
            label="🔗 Close (fill)",
            from_=1,
            to=20,
            orient='horizontal',
            variable=self.close_size,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="➕ Dilate (expand)",
            from_=1,
            to=15,
            orient='horizontal',
            variable=self.dilate_size,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="🏠 Min Area",
            from_=10,
            to=500,
            orient='horizontal',
            variable=self.min_area,
            command=self.on_change).pack(
            fill='x',
            padx=10)
        Scale(
            self.root,
            label="🏠 Max Area",
            from_=1000,
            to=10000,
            orient='horizontal',
            variable=self.max_area,
            command=self.on_change).pack(
            fill='x',
            padx=10)

        button_frame = Frame(self.root)
        button_frame.pack(fill='x', pady=20, padx=10)

        Button(
            button_frame,
            text="💾 Save",
            command=self.save,
            bg='green',
            fg='white',
            font=(
                'Arial',
                11,
                'bold')).pack(
            side='left',
            padx=5)
        Button(
            button_frame,
            text="🔄 Reset",
            command=self.reset,
            bg='orange',
            fg='white',
            font=(
                'Arial',
                11,
                'bold')).pack(
            side='left',
            padx=5)
        Button(
            button_frame,
            text="❌ Close",
            command=self.close,
            bg='red',
            fg='white',
            font=(
                'Arial',
                11,
                'bold')).pack(
            side='right',
            padx=5)

    def on_change(self, value=None):
        """Update parameters in real time"""
        params = {
            'green_lower': [
                self.g_h_min.get(),
                self.g_s_min.get(),
                self.g_v_min.get()],
            'green_upper': [
                self.g_h_max.get(),
                self.g_s_max.get(),
                self.g_v_max.get()],
            'yellow_lower': [
                self.y_h_min.get(),
                100,
                self.y_v_min.get()],
            'yellow_upper': [
                self.y_h_max.get(),
                255,
                self.y_v_max.get()],
            'close_size': self.close_size.get(),
            'dilate_size': self.dilate_size.get(),
            'min_area': self.min_area.get(),
            'max_area': self.max_area.get()}
        self.calibrator.update_params(params)

    def save(self):
        """Save configuration"""
        self.calibrator.save_config()
        print("💾 Configuration saved")

    def reset(self):
        """Reset values"""
        self.g_v_min.set(120)
        self.close_size.set(8)
        self.dilate_size.set(6)
        self.on_change()
        print("🔄 Reset applied")

    def close(self):
        """Close window"""
        self.calibrator.control_active = False
        self.root.destroy()

    def run(self):
        """Run window"""
        self.root.mainloop()


class StaticHSVCalibratorPlus:
    """Advanced HSV calibrator for fragmented stars"""

    def __init__(self):
        self.image_path = Path(__file__).parent.parent / \
            'data/templates/image.png'
        if not self.image_path.exists():
            print(f"❌ Not found {self.image_path}")
            sys.exit(1)

        self.original = cv2.imread(str(self.image_path))
        if self.original is None:
            print("❌ Error loading image")
            sys.exit(1)

        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        print(f"✅ Image: {self.original.shape[1]}x{self.original.shape[0]}")

        self.params = {
            'green_lower': [49, 49, 120],
            'green_upper': [56, 216, 255],
            'yellow_lower': [28, 100, 231],
            'yellow_upper': [29, 255, 238],
            'close_size': 8,
            'dilate_size': 6,
            'min_area': 80,
            'max_area': 4000
        }

        self.control_active = False
        self.info_text = ""
        self.view_mode = 0  # 0=cajas, 1=verde, 2=amarilla

        print("🎨 HSV Calibrator Plus ready")
        print("🎯 Optimized for fragmented stars")

    def update_params(self, new_params):
        """Update parameters from control"""
        self.params.update(new_params)

    def detect_color_boxes(self, color='green'):
        """Detect boxes of a specific color - GLOBAL"""
        if color == 'green':
            lower = np.array(self.params['green_lower'])
            upper = np.array(self.params['green_upper'])
            box_color = (0, 255, 0)
        else:
            lower = np.array(self.params['yellow_lower'])
            upper = np.array(self.params['yellow_upper'])
            box_color = (0, 255, 255)

        mask = cv2.inRange(self.hsv, lower, upper)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.params['close_size'], self.params['close_size']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.params['dilate_size'], self.params['dilate_size']))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['min_area'] <= area <= self.params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio <= 5.0:
                    boxes.append({'x': x, 'y': y, 'w': w, 'h': h,
                                  'area': area, 'color': box_color})

        return boxes, mask

    def process_frame(self):
        """Process frame with detection"""
        green_boxes, green_mask = self.detect_color_boxes('green')
        yellow_boxes, yellow_mask = self.detect_color_boxes('yellow')

        if self.view_mode == 0:
            output = self.original.copy()
            show_boxes = True
        elif self.view_mode == 1:
            output = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            show_boxes = False
        elif self.view_mode == 2:
            output = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            show_boxes = False

        if show_boxes:
            for box in green_boxes:
                cv2.rectangle(
                    output,
                    (box['x'],
                     box['y']),
                    (box['x'] +
                     box['w'],
                        box['y'] +
                        box['h']),
                    box['color'],
                    2)
                cv2.putText(
                    output,
                    f"G:{int(box['area'])}",
                    (box['x'],
                     box['y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box['color'],
                    1)

            for box in yellow_boxes:
                cv2.rectangle(
                    output,
                    (box['x'],
                     box['y']),
                    (box['x'] +
                     box['w'],
                        box['y'] +
                        box['h']),
                    box['color'],
                    2)
                cv2.putText(
                    output,
                    f"Y:{int(box['area'])}",
                    (box['x'],
                     box['y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box['color'],
                    1)

        self.add_overlay(output, len(green_boxes), len(yellow_boxes))
        return output

    def add_overlay(self, img, green_count, yellow_count):
        """On-screen info"""
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (550, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        cv2.putText(
            img,
            f"🟢 VERDES: {green_count} | 🟡 AMARILLAS: {yellow_count}",
            (15,
             35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,
             255,
             255),
            2)

        g_v_min = self.params['green_lower'][2]
        cv2.putText(
            img,
            f"Verde V_min: {g_v_min} | Close: {self.params['close_size']} | Dilate: {self.params['dilate_size']}",
            (15,
             60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,
             255,
             0),
            1)

        modes = ["📦 CAJAS", "🟢 VERDE", "🟡 AMARILLA"]
        cv2.putText(
            img,
            f"Vista: {modes[self.view_mode]}",
            (15,
             80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200,
             200,
             200),
            1)

        cv2.putText(
            img,
            "CTRL: 'c'=Control | 's'=Save | '1'/'2'/'3'=View | 'q'=Exit",
            (15,
             105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150,
             150,
             150),
            1)

    def mouse_callback(self, event, x, y, flags, param):
        """Info of clicked pixel"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < self.hsv.shape[0] and 0 <= x < self.hsv.shape[1]:
                hsv_pixel = self.hsv[y, x]
                bgr_pixel = self.original[y, x]
                print(
                    f"🎯 Pixel ({x},{y}) - HSV: {hsv_pixel} | BGR: {bgr_pixel}")

    def open_control(self):
        """Open control window"""
        if not self.control_active:
            self.control_active = True

            def run_control():
                control = ControlWindow(self)
                control.run()
                self.control_active = False
            threading.Thread(target=run_control, daemon=True).start()
            print("🎛️ Control window opened")

    def save_config(self):
        """Save configuration to file"""
        file_path = Path(__file__).parent.parent / \
            'hsv_ranges_optimized_plus.txt'
        try:
            with open(file_path, 'w') as f:
                f.write("# Optimized HSV Plus - Fragmented Stars\n\n")
                gl = self.params['green_lower']
                gu = self.params['green_upper']
                yl = self.params['yellow_lower']
                yu = self.params['yellow_upper']

                f.write(
                    f"green_lower = np.array([{gl[0]}, {gl[1]}, {gl[2]}])\n")
                f.write(
                    f"green_upper = np.array([{gu[0]}, {gu[1]}, {gu[2]}])\n")
                f.write(
                    f"yellow_lower = np.array([{yl[0]}, {yl[1]}, {yl[2]}])\n")
                f.write(
                    f"yellow_upper = np.array([{yu[0]}, {yu[1]}, {yu[2]}])\n\n")
                f.write(f"# Optimized morphology\n")
                f.write(f"close_size = {self.params['close_size']}\n")
                f.write(f"dilate_size = {self.params['dilate_size']}\n")
                f.write(f"min_area = {self.params['min_area']}\n")
                f.write(f"max_area = {self.params['max_area']}\n")
            print(f"💾 Saved in: {file_path}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def run(self):
        """Run calibrator"""
        print("\n" + "=" * 60)
        print("🎨 HSV CALIBRATOR PLUS")
        print("=" * 60)
        print("🎯 OBJECTIVE: Detect those 2 fragmented green stars")
        print("💡 Initial configuration: Green V_min = 120 (was 229)")
        print("🔧 Aggressive morphology: Close=8, Dilate=6")
        print("")
        print("CONTROLS:")
        print("- 'c': Open control window")
        print("- '1': Original + Boxes view")
        print("- '2': Green Mask view")
        print("- '3': Yellow Mask view")
        print("- 's': Save configuration")
        print("- 'q': Exit")
        print("- Click: Pixel info")
        print("=" * 60)

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
                    print("📦 View: Original + Boxes")
                elif key == ord('2'):
                    self.view_mode = 1
                    print("🟢 View: Green Mask")
                elif key == ord('3'):
                    self.view_mode = 2
                    print("🟡 View: Yellow Mask")

        except KeyboardInterrupt:
            print("\n⛔ Interrupted")
        finally:
            cv2.destroyAllWindows()
            print("👋 Calibrator closed")


def main():
    """Main function"""
    try:
        calibrator = StaticHSVCalibratorPlus()
        calibrator.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
