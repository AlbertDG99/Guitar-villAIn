import cv2
import numpy as np
from mss import mss
from src.utils.config_manager import ConfigManager

drawing = False
ix, iy = -1, -1
roi = None


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback to draw the rectangle."""
    global ix, iy, drawing, roi
    frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(
                "Combo Calibrator - Draw rectangle and press 's'",
                img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        roi = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        print(f"ROI selected: {roi}. Press 's' to confirm and exit.")


def main():
    """Main function for the calibration script."""
    global roi

    print("Starting combo region calibrator...")

    try:
        config = ConfigManager()
        capture_area = config.get_capture_area_config()
        if not capture_area:
            print("Error: Could not get capture area from config.ini.")
            print(
                "Make sure the [CAPTURE] or [calibration] section is properly configured.")
            return
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        return

    window_name = "Combo Calibrator - Draw rectangle and press 's'"
    cv2.namedWindow(window_name)

    with mss() as sct:
        print("\n--- INSTRUCTIONS ---")
        print("1. Draw a rectangle around the combo meter with the mouse.")
        print("2. Once satisfied with the selection, press 's' to save.")
        print("3. Press 'q' to exit without saving.")
        print("---------------------\n")

        while True:
            frame = np.array(sct.grab(capture_area))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            cv2.setMouseCallback(window_name, draw_rectangle, frame)

            if roi:
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(25) & 0xFF

            if key == ord('s'):
                if roi:
                    print("\nRegion saved successfully!")
                    print(
                        "Copy the following line to your config.ini file, within the [COMBO] section:")
                    print(
                        f"combo_region = {{'left': {roi[0]}, 'top': {roi[1]}, 'width': {roi[2]}, 'height': {roi[3]}}}")
                    break
                else:
                    print(
                        "Error: You haven't selected any region. Draw a rectangle first.")

            elif key == ord('q'):
                print("Exiting without saving.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
