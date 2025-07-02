#!/usr/bin/env python3
"""
Score Detector - Detector de Puntuación
=======================================

Módulo para detectar y leer la puntuación del juego usando OCR.
"""

import cv2
import numpy as np
import pytesseract

from src.utils.logger import setup_logger

class ScoreDetector:
    """
    Clase para detectar y gestionar la puntuación del juego.
    Mantiene el estado de la puntuación actual y la actualiza.
    """

    def __init__(self, score_region=None):
        """
        Inicializa el detector de puntuación.
        
        Args:
            score_region (dict, optional): Región de la pantalla donde se encuentra la puntuación.
        """
        self.logger = setup_logger("ScoreDetector")
        self.current_score = 0
        self.region = score_region
        # Configuración de Tesseract:
        # --psm 7: Tratar la imagen como una única línea de texto.
        # -c tessedit_char_whitelist: Aceptar solo estos caracteres.
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789'

    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para mejorar la precisión del OCR.
        La puntuación suele ser texto blanco/brillante sobre fondo oscuro.
        """
        # 1. Convertir a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Aplicar umbralización (thresholding) para aislar el texto
        # THRESH_BINARY_INV: Pone los píxeles brillantes en negro y los oscuros en blanco.
        # El OCR funciona mejor con texto negro sobre fondo blanco.
        # El valor 180 es un buen punto de partida, puede requerir ajuste.
        _, thresh_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Opcional: Aplicar un filtro de desenfoque para suavizar el ruido
        # thresh_image = cv2.medianBlur(thresh_image, 3)

        return thresh_image

    def update_score(self, frame: np.ndarray) -> int:
        """
        Toma el frame completo del juego, recorta la región de la puntuación,
        detecta el número y actualiza la puntuación interna.
        """
        if self.region is None or frame is None or frame.size == 0:
            return self.current_score

        x, y, w, h = self.region['x'], self.region['y'], self.region['width'], self.region['height']
        image = frame[y:y+h, x:x+w]

        if image.size == 0:
            return self.current_score

        try:
            # 1. Preprocesar la imagen
            processed_image = self._preprocess_image_for_ocr(image)

            # 2. Usar Pytesseract para extraer el texto
            extracted_text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)

            # 3. Limpiar y convertir el resultado
            # Eliminar espacios, saltos de línea o cualquier carácter no numérico
            cleaned_text = "".join(filter(str.isdigit, extracted_text))

            if cleaned_text:  # Regla #2: La puntuación siempre debe ser un número
                new_score = int(cleaned_text)

                # Regla #1: La puntuación no puede bajar
                if new_score > self.current_score:
                    self.logger.info("Puntuación actualizada: %d -> %d", self.current_score, new_score)
                    self.current_score = new_score

        except pytesseract.TesseractNotFoundError:
            self.logger.error("Tesseract no está instalado o no se encuentra en el PATH del sistema.")
            self.logger.error("Por favor, instala Tesseract-OCR y asegúrate de que se puede llamar desde la terminal.")
            # Detener el programa si Tesseract no está disponible.
            raise
        except Exception as e:
            self.logger.error("Error al detectar la puntuación: %s", e)
        
        return self.current_score 