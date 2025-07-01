#!/usr/bin/env python3
"""
Guitar Hero IA - Interfaz Principal Unificada
==============================================

Sistema completo de Guitar Hero con IA controlado por línea de comandos.
Incluye calibración, diagnóstico, entrenamiento y ejecución.
"""

import argparse
import sys
import time
from pathlib import Path
import torch

# Añadir src al path si no está
# Este bloque debe ir antes de las importaciones del proyecto
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from src.window_calibrator import WindowCalibrator
from src.guitar_hero_hotkeys import GuitarHeroHotkeySystem
from src.utils.logger import setup_logger
from src.monitor_setup import MonitorSetup
# from src.detection_debugger import DetectionDebugger  # TODO: Reimplementar
# from src.reinforcement_trainer import ReinforcementTrainer  # TODO: Reimplementar
from src.ai.dqn_agent import DQNAgent
from src.utils.config_manager import ConfigManager
from src.core.note_detector import NoteDetector
from src.core.input_controller import InputController


def show_banner():
    """Mostrar banner del sistema"""
    print("🎸" + "=" * 58 + "🎸")
    print("🎵            GUITAR HERO IA - SISTEMA COMPLETO            🎵")
    print("🤖         Aprendizaje Reforzado + Computer Vision         🤖")
    print("🎸" + "=" * 58 + "🎸")
    print()


def show_main_menu():
    """Mostrar menú principal completo"""
    print("📋 MENÚ PRINCIPAL")
    print("=" * 40)
    print()
    print("🔧 CONFIGURACIÓN Y CALIBRACIÓN:")
    print("  1. 📏 Calibrar ventana del juego")
    print("  2. 🖥️  Configurar monitores")
    print()
    print("🔍 DIAGNÓSTICO Y DEPURACIÓN:")
    print("  3. 🔬 Diagnosticar detección de notas")
    print("  4. 📊 Analizar imagen específica")
    print("  5. 🧪 Test de sistema completo")
    print()
    print("🧠 INTELIGENCIA ARTIFICIAL:")
    print("  6. ❌ Entrenar nuevo modelo IA (deshabilitado)")
    print("  7. ❌ Continuar entrenamiento (deshabilitado)")
    print("  8. ❌ Evaluar modelo existente (deshabilitado)")
    print("  9. ❌ Demo con IA entrenada (deshabilitado)")
    print()
    print("🚀 EJECUCIÓN:")
    print("  10. 🚀 EJECUCIÓN CON HOTKEYS")
    print("  11. 🤖 Ejecutar solo IA automática")
    print()
    print("ℹ️  INFORMACIÓN:")
    print("  12. 📖 Mostrar ayuda detallada")
    print("  13. 📊 Estado del sistema")
    print("  0.  🚪 Salir")
    print()


def calibrate_window():
    """Ejecutar calibración de ventana"""
    print("📏 CALIBRACIÓN DE VENTANA")
    print("=" * 30)

    calibrator = WindowCalibrator()
    success = calibrator.calibrate()

    if success:
        print("\n✅ Calibración completada exitosamente")
        print("\n📋 Siguiente paso: Diagnosticar detección (opción 3)")
    else:
        print("\n❌ Error en calibración")

    input("\nPresiona Enter para continuar...")


def setup_monitors():
    """Configurar monitores"""
    print("🖥️ CONFIGURACIÓN DE MONITORES")
    print("=" * 35)

    try:
        setup = MonitorSetup()
        success = setup.setup_dual_monitor()
        if success:
            print("✅ Configuración de monitores completada")
        else:
            print("❌ Error en configuración de monitores")
    except ImportError:
        print("❌ Error: No se pudo importar MonitorSetup. "
              "Asegúrate de que 'screeninfo' está instalado.")
    except Exception as e:
        print(f"❌ Error inesperado en configuración: {e}")

    input("\nPresiona Enter para continuar...")


def diagnose_detection():
    """Ejecutar diagnóstico de detección"""
    print("🔬 DIAGNÓSTICO DE DETECCIÓN DE NOTAS")
    print("=" * 40)
    print()
    print("❌ FUNCIÓN TEMPORALMENTE DESHABILITADA")
    print("El módulo de diagnóstico está siendo reimplementado.")
    print("Por favor usa el sistema principal con hotkeys (opción 10).")
    print()
    input("\nPresiona Enter para continuar...")


def analyze_image():
    """Analizar imagen específica"""
    print("📊 ANÁLISIS DE IMAGEN ESPECÍFICA")
    print("=" * 35)
    print()
    print("❌ FUNCIÓN TEMPORALMENTE DESHABILITADA")
    print("El módulo de análisis está siendo reimplementado.")
    print()
    input("\nPresiona Enter para continuar...")


def test_system():
    """Test completo del sistema"""
    print("🧪 TEST DE SISTEMA COMPLETO")
    print("=" * 30)

    try:
        print("🔍 Verificando componentes...")

        # Test calibración
        print("📏 Test calibración: ", end="")
        calibrator = WindowCalibrator()
        region = calibrator.get_capture_region()
        if region:
            print("✅ OK")
        else:
            print("❌ FALLO")

        # Test componentes principales
        print("🧠 Test agente IA: ", end="")
        config = ConfigManager()
        agent = DQNAgent(config)
        del agent  # Liberar memoria
        print("✅ OK")
        
        print("👁️ Test detector notas: ", end="")
        detector = NoteDetector(config)
        del detector # Liberar memoria
        print("✅ OK")
        
        print("⌨️ Test controlador input: ", end="")
        controller = InputController(config)
        del controller # Liberar memoria
        print("✅ OK")
        
        print("\n✅ Sistema completamente funcional")

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Error de importación: {e}. "
              "Asegúrate de que todas las dependencias están instaladas.")
    except Exception as e:
        print(f"❌ Error inesperado en test del sistema: {e}")

    input("\nPresiona Enter para continuar...")


def train_new_model():
    """Entrenar nuevo modelo de IA"""
    print("🎓 ENTRENAMIENTO DE NUEVO MODELO")
    print("=" * 35)
    print()
    print("❌ FUNCIÓN TEMPORALMENTE DESHABILITADA")
    print("El módulo de entrenamiento está siendo reimplementado.")
    print()
    input("\nPresiona Enter para continuar...")


def continue_training():
    """Continuar entrenamiento existente"""
    print("📚 CONTINUAR ENTRENAMIENTO")
    print("=" * 30)

    try:
        trainer = ReinforcementTrainer()

        model_path = input("Ruta del modelo a continuar: ").strip()
        if model_path and Path(model_path).exists():
            episodes = int(input("Episodios adicionales (default: 50): ") or "50")
            trainer.load_pretrained_model(model_path)
            trainer.start_training(episodes)
        else:
            print(f"❌ No se pudo encontrar el modelo en: {model_path}")

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Error importando entrenador: {e}")
    except ValueError:
        print("❌ Error: El número de episodios debe ser un entero.")
    except Exception as e:
        print(f"❌ Error inesperado en entrenamiento: {e}")

    input("\nPresiona Enter para continuar...")


def evaluate_model():
    """Evaluar modelo existente"""
    print("🧪 EVALUACIÓN DE MODELO")
    print("=" * 25)

    try:
        trainer = ReinforcementTrainer()

        model_path = input("Ruta del modelo a evaluar: ").strip()
        if model_path and Path(model_path).exists():
            episodes = int(input("Número de episodios de evaluación (default: 10): ") or "10")
            trainer.load_pretrained_model(model_path)
            trainer.evaluate_model(episodes)
        else:
            print(f"❌ No se pudo encontrar el modelo en: {model_path}")

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Error importando componentes de evaluación: {e}")
    except ValueError:
        print("❌ Error: El número de episodios debe ser un entero.")
    except Exception as e:
        print(f"❌ Error inesperado durante la evaluación: {e}")

    input("\nPresiona Enter para continuar...")


def demo_mode():
    """Modo demo con IA entrenada"""
    print("🎮 MODO DEMO CON IA")
    print("=" * 20)

    try:
        trainer = ReinforcementTrainer()

        model_path = input("Ruta del modelo para el modo demo: ").strip()

        if model_path and Path(model_path).exists():
            print("\n🚀 Iniciando modo demo...")
            print("   La IA jugará automáticamente.")
            print("   Presiona Ctrl+C en la terminal para detener.")
            trainer.load_pretrained_model(model_path)
            trainer.evaluate_model(episodes=1000)  # Bucle largo para demo
        else:
            print(f"❌ No se pudo encontrar el modelo en: {model_path}")

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Error importando componentes para el modo demo: {e}")
    except Exception as e:
        print(f"❌ Error inesperado en modo demo: {e}")

    input("\nPresiona Enter para continuar...")


def run_with_hotkeys():
    """Ejecutar el sistema con control por hotkeys."""
    print("🚀 EJECUCIÓN CON HOTKEYS")
    print("=" * 30)
    print("🚀 Iniciando sistema con Hotkeys...")
    print("   El sistema está activo y esperando tus comandos.")
    print("   Consulta la ayuda (opción 12) para ver las teclas.")
    print("   Presiona Ctrl+C en la terminal para salir.")

    try:
        hotkey_system = GuitarHeroHotkeySystem()
        hotkey_system.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Saliendo del modo Hotkeys.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

    input("\nPresiona Enter para volver al menú...")


def run_ai_only():
    """Ejecutar solo IA automática"""
    print("🤖 EJECUTAR SOLO IA AUTOMÁTICA")
    print("=" * 35)
    print("🤖 Iniciando modo IA automática...")
    print("   La IA controlará el juego sin intervención.")
    print("   Presiona Ctrl+C en la terminal para detener.")

    try:
        trainer = ReinforcementTrainer()
        model_path = input("Ruta del modelo para la IA: ").strip()

        if model_path and Path(model_path).exists():
            trainer.load_pretrained_model(model_path)
            trainer.ai_agent.set_eval_mode()
            # Bucle de juego automático
            while True:
                state = trainer._reset_episode()
                done = False
                while not done:
                    action = trainer.ai_agent.get_action(state, training=False)
                    state, _, done, _ = trainer._step(action)
                    time.sleep(0.01) # Pequeña pausa
        else:
            print(f"❌ No se pudo encontrar el modelo en: {model_path}")

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Error importando componentes de IA: {e}")
    except KeyboardInterrupt:
        print("\n👋 Saliendo del modo IA automática.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

    input("\nPresiona Enter para continuar...")


def show_help():
    """Mostrar ayuda detallada"""
    print("📖 AYUDA DETALLADA")
    print("=" * 20)
    print()
    print("🔧 FLUJO RECOMENDADO:")
    print("1. Calibrar ventana (opción 1)")
    print("2. Diagnosticar detección (opción 3)")
    print("3. Entrenar modelo IA (opción 6)")
    print("4. Ejecutar sistema (opción 10)")
    print()
    print("🎯 DESCRIPCIÓN DE OPCIONES:")
    print()
    print("📏 CALIBRACIÓN:")
    print("   Establece la región exacta donde está el juego")
    print("   Debe hacerse antes de cualquier detección")
    print()
    print("🔬 DIAGNÓSTICO:")
    print("   Herramienta visual para verificar detección")
    print("   Muestra qué notas ve el sistema en tiempo real")
    print("   Permite ajustar rangos de color HSV")
    print()
    print("🧠 ENTRENAMIENTO IA:")
    print("   Usa Deep Q-Learning para aprender a jugar")
    print("   Requiere calibración previa correcta")
    print("   Guarda modelos cada 10 episodios")
    print()
    print("🚀 EJECUCIÓN:")
    print("   Sistema completo con hotkeys globales")
    print("   No pausa el juego al cambiar ventanas")
    print("   Funciona en segundo plano")
    print()

    input("\nPresiona Enter para continuar...")


def show_system_status():
    """Mostrar estado del sistema"""
    print("📊 ESTADO DEL SISTEMA")
    print("=" * 25)

    try:
        # Verificar calibración
        print("📏 Calibración: ", end="")
        calibrator = WindowCalibrator()
        region = calibrator.get_capture_region()
        if region:
            print(f"✅ OK ({region['width']}x{region['height']})")
        else:
            print("❌ NO CALIBRADO")

        # Verificar configuración
        print("⚙️ Configuración: ", end="")
        config = ConfigManager()
        if config:
            print("✅ OK")

        # Verificar PyTorch/CUDA
        print("🔥 PyTorch/CUDA: ", end="")
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ OK ({gpu_name})")
            else:
                print("⚠️ Solo CPU")
        except ImportError:
            print("❌ PyTorch no instalado")

        # Verificar modelos entrenados
        print("🤖 Modelos IA: ", end="")
        models_dir = Path("models")
        if models_dir.exists():
            models = list(models_dir.glob("*.pth"))
            print(f"✅ {len(models)} modelo(s) encontrado(s)")
        else:
            print("❌ Sin modelos entrenados")

        # Mostrar archivos de log
        print("📜 Logs: ", end="")
        logs_dir = Path("logs")
        if logs_dir.exists():
            logs = list(logs_dir.glob("*.log"))
            print(f"✅ {len(logs)} archivo(s) de log")
        else:
            print("⚠️ Sin logs")

        print()
        print("💡 RECOMENDACIONES:")
        if not region:
            print("   • Ejecuta calibración (opción 1)")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            print("   • Entrena un modelo IA (opción 6)")
        print("   • Usa diagnóstico para verificar detección (opción 3)")

    except Exception as e:
        print(f"❌ Error verificando estado: {e}")

    input("\nPresiona Enter para continuar...")


def main():
    """Función principal que ejecuta el menú interactivo."""
    # Desactivar el banner de Pygame si se usa
    # os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    parser = argparse.ArgumentParser(
        description="Guitar Hero IA - Sistema Completo",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--skip-menu",
        action="store_true",
        help="Omitir el menú principal y ejecutar una acción directamente."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=[
            'calibrate', 'monitors', 'diagnose', 'analyze',
            'test', 'train', 'continue', 'evaluate', 'demo',
            'hotkeys', 'ai_only', 'help', 'status'
        ],
        help="Acción a ejecutar si se omite el menú."
    )
    args = parser.parse_args()

    menu_actions = {
        '1': calibrate_window,
        '2': setup_monitors,
        '3': diagnose_detection,
        '4': analyze_image,
        '5': test_system,
        '6': train_new_model,
        '7': continue_training,
        '8': evaluate_model,
        '9': demo_mode,
        '10': run_with_hotkeys,
        '11': run_ai_only,
        '12': show_help,
        '13': show_system_status,
    }

    if args.skip_menu and args.action:
        action_map = {
            'calibrate': calibrate_window, 'monitors': setup_monitors,
            'diagnose': diagnose_detection, 'analyze': analyze_image,
            'test': test_system, 'train': train_new_model,
            'continue': continue_training, 'evaluate': evaluate_model,
            'demo': demo_mode, 'hotkeys': run_with_hotkeys,
            'ai_only': run_ai_only, 'help': show_help, 'status': show_system_status
        }
        action_func = action_map.get(args.action)
        if action_func:
            action_func()
        else:
            print(f"Acción desconocida: {args.action}")
        return

    show_banner()
    while True:
        show_main_menu()
        choice = input("Selecciona una opción (0-13): ").strip()

        if choice == '0':
            print("👋 ¡Hasta luego!")
            break

        action = menu_actions.get(choice)
        if action:
            try:
                action()
            except Exception as e:
                print(f"❌ Ocurrió un error inesperado en la opción '{choice}': {e}")
                input("\nPresiona Enter para continuar...")
        else:
            print("❌ Opción no válida. Por favor, elige un número del 0 al 13.")
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario. ¡Adiós!")
        sys.exit(0)
    except Exception as e:
        # Captura final para errores no manejados
        setup_logger("MAIN_ERROR").critical("Error fatal no capturado: %s", e, exc_info=True)
        print(f"\n\n❌ ERROR FATAL: {e}")
        print("Se ha registrado un error crítico. Revisa 'logs/error.log' para más detalles.")
        sys.exit(1)
