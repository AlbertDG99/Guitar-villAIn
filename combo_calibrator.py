import cv2
import numpy as np
from mss import mss
from src.utils.config_manager import ConfigManager

# Variables globales para el dibujo del rectángulo
drawing = False
ix, iy = -1, -1
roi = None

def draw_rectangle(event, x, y, flags, param):
    """Callback del ratón para dibujar el rectángulo."""
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
            cv2.imshow("Calibrador de Combo - Dibuja el rectangulo y pulsa 's'", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        # Guardar las coordenadas del ROI (x, y, w, h)
        roi = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        print(f"ROI seleccionado: {roi}. Pulsa 's' para confirmar y salir.")

def main():
    """Función principal para el script de calibración."""
    global roi
    
    print("Iniciando calibrador de la región del combo...")
    
    try:
        config = ConfigManager()
        capture_area = config.get_capture_area_config()
        if not capture_area:
            print("Error: No se pudo obtener el área de captura desde config.ini.")
            print("Asegúrate de que la sección [CAPTURE] o [calibration] está bien configurada.")
            return
    except Exception as e:
        print(f"Error al inicializar la configuración: {e}")
        return

    window_name = "Calibrador de Combo - Dibuja el rectangulo y pulsa 's'"
    cv2.namedWindow(window_name)
    
    with mss() as sct:
        print("\n--- INSTRUCCIONES ---")
        print("1. Dibuja un rectángulo alrededor del medidor de combo con el ratón.")
        print("2. Una vez satisfecho con la selección, pulsa la tecla 's' para guardar.")
        print("3. Pulsa 'q' para salir sin guardar.")
        print("---------------------\n")
        
        # Bucle principal para la captura y visualización
        while True:
            frame = np.array(sct.grab(capture_area))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Pasar el frame actual como parámetro al callback
            cv2.setMouseCallback(window_name, draw_rectangle, frame)
            
            if roi:
                 # Dibuja el rectangulo final si existe
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(25) & 0xFF
            
            if key == ord('s'):
                if roi:
                    print("\n¡Región guardada exitosamente!")
                    print("Copia la siguiente línea en tu fichero config.ini, dentro de la sección [COMBO]:")
                    print(f"combo_region = {{'left': {roi[0]}, 'top': {roi[1]}, 'width': {roi[2]}, 'height': {roi[3]}}}")
                    break
                else:
                    print("Error: No has seleccionado ninguna región. Dibuja un rectángulo primero.")
            
            elif key == ord('q'):
                print("Saliendo sin guardar.")
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 