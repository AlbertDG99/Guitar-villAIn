import cv2
import numpy as np
from typing import List, Tuple

def display_actions(window_name, action_vector, key_names):
    """
    Muestra el vector de acciones en una ventana de OpenCV.

    Args:
        window_name (str): El nombre de la ventana de OpenCV.
        action_vector (list or np.array): El vector de acciones (0: nada, 1: pulsar, 2: soltar).
        key_names (list): La lista de nombres de las teclas correspondientes.
    """
    width = 600
    height_per_key = 50
    header_height = 40
    height = header_height + len(key_names) * height_per_key

    # Crear una imagen en blanco
    img = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)

    # Título
    cv2.putText(img, "AI Action Monitor", (width // 2 - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, (action, key_name) in enumerate(zip(action_vector, key_names)):
        y_pos = header_height + i * height_per_key
        
        # Color y texto de la acción
        if action == 1:  # Pulsar
            color = (0, 255, 0)  # Verde
            text = "PRESSING"
        elif action == 2:  # Soltar
            color = (255, 100, 100)  # Rojo claro
            text = "RELEASING"
        else:  # Nada
            color = (80, 80, 80)  # Gris oscuro
            text = "IDLE"
            
        # Dibujar rectángulos y texto para cada tecla
        cv2.rectangle(img, (10, y_pos + 5), (width - 10, y_pos + height_per_key - 5), color, -1)
        
        cv2.putText(img, f"Key: {key_name.upper()}", (20, y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
        cv2.putText(img, text, (width - 150, y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow(window_name, img)
    cv2.waitKey(1)

def is_note_active(frame: np.ndarray, polygon: List[Tuple[int, int]]) -> bool:
    """
    Comprueba si hay alguna nota activa dentro de un polígono de carril.
    Esta es una implementación simplificada y puede ser reemplazada por una
    más compleja que use detección de color y contornos.
    """
    if frame is None or not polygon:
        return False
        
    # Crear una máscara para el polígono
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    
    # Calcular la media de color dentro de la máscara.
    # Si es suficientemente brillante, consideramos que hay una nota.
    # Este es un método muy simple y puede necesitar ajustes.
    mean_val = cv2.mean(frame, mask=mask)
    
    # Sumar los valores BGR y comprobar si superan un umbral.
    # Un valor de 50 es un umbral bajo, asumiendo que el carril es oscuro
    # y las notas son brillantes.
    brightness_threshold = 50 
    
    if sum(mean_val[:3]) > brightness_threshold:
        return True
        
    return False 