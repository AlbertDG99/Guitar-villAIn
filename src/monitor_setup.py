"""
Monitor Setup - Configurador de Monitores
==========================================

Configuración específica para setup de doble monitor con Guitar Hero IA.
"""

import time
from typing import Tuple, Dict

import mss
import cv2
import numpy as np

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger



class MonitorSetup:  # pylint: disable=too-many-instance-attributes
    """Configurador de monitores para Guitar Hero IA"""

    def __init__(self):
        self.logger = setup_logger('MonitorSetup')
        self.config = ConfigManager()

        # Detectar monitores
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

        self.logger.info("MonitorSetup inicializado")
        self._log_monitor_info()

    def _log_monitor_info(self):
        """Mostrar información de monitores detectados"""
        self.logger.info("🖥️  MONITORES DETECTADOS:")
        self.logger.info("=" * 40)

        for i, monitor in enumerate(self.monitors):
            if i == 0:
                self.logger.info(f"Monitor {i}: VIRTUAL TOTAL - {monitor['width']}x{monitor['height']}")
            else:
                self.logger.info(f"Monitor {i}: {monitor['width']}x{monitor['height']} en ({monitor['left']}, {monitor['top']})")

        self.logger.info("=" * 40)

    def select_monitor_interactive(self) -> int:
        """Seleccionar monitor de forma interactiva"""
        print("\n🖥️  SELECCIÓN DE MONITOR PARA EL JUEGO")
        print("=" * 45)

        for i, monitor in enumerate(self.monitors[1:], 1):  # Saltar monitor 0 (virtual)
            side = "IZQUIERDO" if monitor['left'] == 0 else "DERECHO"
            print(f"{i}. Monitor {side}: {monitor['width']}x{monitor['height']}")

        while True:
            try:
                choice = input(f"\n➤ ¿En qué monitor está el juego? (1-{len(self.monitors)-1}): ").strip()
                monitor_id = int(choice)

                if 1 <= monitor_id <= len(self.monitors) - 1:
                    return monitor_id
                else:
                    print("❌ Opción inválida. Intenta de nuevo.")

            except ValueError:
                print("❌ Por favor ingresa un número válido.")

    def configure_monitor(self, monitor_id: int) -> bool:
        """Configurar monitor seleccionado"""
        if monitor_id >= len(self.monitors):
            self.logger.error(f"Monitor {monitor_id} no existe")
            return False

        monitor = self.monitors[monitor_id]

        # Actualizar configuración
        self.config.set('CAPTURE', 'target_monitor', str(monitor_id))
        self.config.set('CAPTURE', 'monitor_left', str(monitor['left']))
        self.config.set('CAPTURE', 'monitor_top', str(monitor['top']))
        self.config.set('CAPTURE', 'monitor_width', str(monitor['width']))
        self.config.set('CAPTURE', 'monitor_height', str(monitor['height']))

        # Configurar región de juego inicial (pantalla completa del monitor)
        self.config.set('CAPTURE', 'game_left', str(monitor['left']))
        self.config.set('CAPTURE', 'game_top', str(monitor['top']))
        self.config.set('CAPTURE', 'game_width', str(monitor['width']))
        self.config.set('CAPTURE', 'game_height', str(monitor['height']))

        side = "IZQUIERDO" if monitor['left'] == 0 else "DERECHO"
        self.logger.info(f"✅ Monitor {side} configurado como objetivo")
        self.logger.info(f"   Resolución: {monitor['width']}x{monitor['height']}")
        self.logger.info(f"   Posición: ({monitor['left']}, {monitor['top']})")

        return True

    def test_capture(self, monitor_id: int, duration: float = 5.0) -> bool:
        """Probar captura en el monitor seleccionado"""
        if monitor_id >= len(self.monitors):
            return False

        monitor = self.monitors[monitor_id]

        self.logger.info(f"🎯 Probando captura en monitor {monitor_id} por {duration} segundos...")
        self.logger.info("   (Se guardará screenshot de prueba)")

        try:
            # Capturar screenshot
            screenshot = self.sct.grab(monitor)

            # Convertir a numpy array
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Guardar screenshot de prueba
            timestamp = int(time.time())
            filename = f"screenshots/monitor_{monitor_id}_test_{timestamp}.png"
            cv2.imwrite(filename, frame)

            # Mostrar información
            side = "IZQUIERDO" if monitor['left'] == 0 else "DERECHO"
            self.logger.info(f"✅ Captura exitosa del monitor {side}")
            self.logger.info(f"   Tamaño: {frame.shape[1]}x{frame.shape[0]} pixels")
            self.logger.info(f"   Screenshot guardado: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error en captura: {e}")
            return False

    def create_calibration_overlay(self, monitor_id: int):
        """Crear overlay de calibración en el monitor seleccionado"""
        if monitor_id >= len(self.monitors):
            return

        monitor = self.monitors[monitor_id]

        try:
            # Crear imagen de calibración
            img = np.zeros((monitor['height'], monitor['width'], 3), dtype=np.uint8)

            # Dibujar marco
            cv2.rectangle(img, (0, 0), (monitor['width']-1, monitor['height']-1), (0, 255, 0), 5)

            # Dibujar líneas de carriles (estimadas)
            lane_width = monitor['width'] // 6
            for i in range(1, 6):
                x = i * lane_width
                cv2.line(img, (x, 0), (x, monitor['height']), (255, 0, 0), 2)

            # Añadir texto
            cv2.putText(img, f"MONITOR {monitor_id} - GUITAR HERO IA",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(img, "Presiona F10 para calibrar area de juego",
                       (50, monitor['height'] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Mostrar ventana de calibración
            window_name = f"Calibración Monitor {monitor_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, monitor['left'], monitor['top'])
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, img)

            self.logger.info("📺 Overlay de calibración mostrado")
            self.logger.info("   Presiona cualquier tecla para cerrar")

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            self.logger.error(f"Error creando overlay: {e}")

    def setup_dual_monitor(self) -> bool:
        """Configuración completa para doble monitor"""
        print("\n🎮 CONFIGURACIÓN PARA DOBLE MONITOR")
        print("=" * 45)
        print("📌 IMPORTANTE:")
        print("   • El script funcionará en monitor secundario")
        print("   • El juego debe estar en monitor principal")
        print("   • Usaremos hotkeys para evitar cambios de ventana")
        print()

        # Seleccionar monitor
        monitor_id = self.select_monitor_interactive()

        # Configurar monitor
        if not self.configure_monitor(monitor_id):
            return False

        print(f"\n🎯 Monitor {monitor_id} seleccionado")
        print("⏳ Probando captura...")

        # Probar captura
        if not self.test_capture(monitor_id):
            return False

        print("\n✅ Configuración completada exitosamente!")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. 🎮 Inicia el juego en el monitor configurado")
        print("2. 🚀 Ejecuta: python src/guitar_hero_hotkeys.py")
        print("3. ⌨️  Usa F10 para calibrar área específica del juego")
        print("4. 🎯 Usa F9 para iniciar/detener la IA")

        return True


def main():
    """Función principal"""
    setup = MonitorSetup()

    if setup.setup_dual_monitor():
        print("\n🎉 ¡Configuración de doble monitor completada!")
    else:
        print("\n❌ Error en la configuración")


if __name__ == "__main__":
    main()
