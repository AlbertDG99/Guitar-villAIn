import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict

class ComboDetector:
    """
    Detecta el multiplicador de combo a partir de una región de la pantalla usando OCR,
    con lógica para evitar lecturas erróneas.
    """
    def __init__(self, config_region: Optional[Dict[str, int]]):
        """
        Inicializa el detector de combo.

        Args:
            config_region (Optional[Dict[str, int]]): 
                El diccionario con la región del combo: {'x', 'y', 'width', 'height'}.
                Si es None, el detector estará deshabilitado.
        """
        if config_region:
            self.x = config_region['x']
            self.y = config_region['y']
            self.width = config_region['width']
            self.height = config_region['height']
            self.enabled = True
        else:
            self.enabled = False
            self.x = self.y = self.width = self.height = 0
            
        # Configuración de Tesseract para leer solo números
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        self.reset()

    def reset(self):
        """Resetea el estado interno del detector."""
        self.last_valid_combo = 1

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para mejorar la precisión del OCR.
        """
        if image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale_factor = 3
        resized = cv2.resize(gray, (self.width * scale_factor, self.height * scale_factor), interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def detect(self, frame: np.ndarray) -> int:
        """
        Analiza el frame, lee el texto del combo y devuelve el multiplicador numérico.

        Returns:
            int: El multiplicador de combo (1, 2, 3, 4). Devuelve 1 si no hay combo o no se detecta nada.
        """
        if not self.enabled or frame is None:
            return self.last_valid_combo

        roi = frame[self.y:self.y + self.height, self.x:self.x + self.width]
        if roi.size == 0:
            return self.last_valid_combo
            
        processed_roi = self._preprocess_for_ocr(roi)
        
        try:
            text = pytesseract.image_to_string(processed_roi, config=self.tesseract_config).strip()
            
            # Si no se lee nada, mantenemos el último combo conocido.
            if not text:
                return self.last_valid_combo

            nums = "".join(filter(str.isdigit, text))
            
            if nums:
                new_combo = int(nums)
                
                # Regla de seguridad: Ignorar saltos irreales en el combo
                if new_combo > self.last_valid_combo + 10:
                    return self.last_valid_combo
                
                self.last_valid_combo = new_combo
                return new_combo
            else:
                # Se leyó texto pero no eran números (ej. "x"), mantener el último valor.
                return self.last_valid_combo
            
        except Exception:
            return self.last_valid_combo

    def get_combo_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Devuelve la ROI del combo para depuración."""
        if not self.enabled:
            return None
        return frame[self.y:self.y + self.height, self.x:self.x + self.width] 