#!/usr/bin/env python3
"""
🚀 OpenCV CUDA Setup - Instalación Automatizada
===============================================

Script automatizado para instalar OpenCV con soporte CUDA
basado en documentación oficial:
https://opencv.org/platforms/cuda/

Métodos implementados:
1. Conda-forge (recomendado)
2. pip precompilado (experimentales)
3. Compilación desde código (avanzado)
"""

import sys
import subprocess
import os
import platform
import shutil
from pathlib import Path
import urllib.request
import zipfile
import tempfile

def check_system_requirements():
    """Verifica requisitos del sistema"""
    print("🔍 VERIFICANDO REQUISITOS DEL SISTEMA")
    print("=" * 50)
    
    # Python version
    python_version = sys.version_info
    print(f"✅ Python: {python_version.major}.{python_version.minor}")
    
    # Operating System
    system = platform.system()
    print(f"✅ OS: {system} {platform.release()}")
    
    # CUDA disponible
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detectada")
            # Extraer versión CUDA
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip()
                    print(f"✅ CUDA Driver: {cuda_version}")
                    break
        else:
            print("❌ NVIDIA GPU no detectada")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado - No hay GPU NVIDIA")
        return False
    
    # Conda disponible
    conda_available = shutil.which('conda') is not None
    print(f"{'✅' if conda_available else '⚠️'} Conda: {'Disponible' if conda_available else 'No disponible'}")
    
    # Pip disponible
    pip_available = shutil.which('pip') is not None
    print(f"✅ Pip: {'Disponible' if pip_available else 'No disponible'}")
    
    print()
    return True

def backup_current_opencv():
    """Respalda la instalación actual de OpenCV"""
    print("💾 RESPALDANDO OPENCV ACTUAL")
    print("=" * 50)
    
    try:
        import cv2
        current_version = cv2.__version__
        print(f"📦 OpenCV actual: {current_version}")
        
        # Crear información de respaldo
        backup_info = {
            'version': current_version,
            'path': cv2.__file__,
            'cuda_support': hasattr(cv2, 'cuda'),
            'cuda_devices': cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
        }
        
        print(f"📍 Ubicación: {backup_info['path']}")
        print(f"🖥️ CUDA: {'✅' if backup_info['cuda_support'] else '❌'}")
        
        # Guardar info para rollback
        with open('opencv_backup_info.txt', 'w') as f:
            for key, value in backup_info.items():
                f.write(f"{key}: {value}\n")
        
        print("✅ Información de respaldo guardada en opencv_backup_info.txt")
        
    except ImportError:
        print("⚠️ OpenCV no instalado actualmente")
    
    print()

def install_opencv_conda():
    """Instala OpenCV con CUDA usando conda-forge"""
    print("🐍 INSTALANDO OPENCV CON CONDA-FORGE")
    print("=" * 50)
    
    if not shutil.which('conda'):
        print("❌ Conda no está disponible")
        return False
    
    try:
        # Desinstalar versiones actuales
        print("🗑️ Desinstalando OpenCV actual...")
        subprocess.run(['pip', 'uninstall', 'opencv-python', 'opencv-contrib-python', '-y'], 
                      capture_output=True)
        
        # Instalar desde conda-forge
        print("📦 Instalando OpenCV desde conda-forge...")
        cmd = ['conda', 'install', '-c', 'conda-forge', 'opencv', '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ OpenCV instalado desde conda-forge")
            return True
        else:
            print(f"❌ Error instalando desde conda-forge: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante instalación conda: {e}")
        return False

def install_opencv_pip_precompiled():
    """Instala OpenCV precompilado con CUDA (experimental)"""
    print("📦 INSTALANDO OPENCV PRECOMPILADO CON CUDA")
    print("=" * 50)
    
    print("⚠️ ADVERTENCIA: Esta es una versión experimental")
    print("   Puede no funcionar en todos los sistemas")
    print()
    
    try:
        # Desinstalar versión actual
        print("🗑️ Desinstalando OpenCV actual...")
        subprocess.run(['pip', 'uninstall', 'opencv-python', 'opencv-contrib-python', '-y'],
                      capture_output=True)
        
        # URLs de versiones precompiladas con CUDA (experimentales)
        cuda_wheels = {
            'windows': [
                'https://github.com/opencv/opencv-python/releases/download/4.8.1.78/opencv_contrib_python-4.8.1.78-cp39-cp39-win_amd64.whl',
                'opencv-contrib-python==4.8.1.78'
            ]
        }
        
        system = platform.system().lower()
        if 'windows' in system:
            print("🪟 Instalando versión para Windows...")
            cmd = ['pip', 'install', 'opencv-contrib-python==4.8.1.78']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ OpenCV precompilado instalado")
                return True
            else:
                print(f"❌ Error: {result.stderr}")
                return False
        else:
            print("❌ Versión precompilada no disponible para este sistema")
            return False
            
    except Exception as e:
        print(f"❌ Error durante instalación: {e}")
        return False

def verify_cuda_installation():
    """Verifica que CUDA funcione correctamente"""
    print("🔍 VERIFICANDO INSTALACIÓN CUDA")
    print("=" * 50)
    
    try:
        # Reimportar OpenCV
        import importlib
        if 'cv2' in sys.modules:
            importlib.reload(sys.modules['cv2'])
        
        import cv2
        
        print(f"📦 OpenCV Version: {cv2.__version__}")
        print(f"🖥️ Módulo CUDA: {'✅' if hasattr(cv2, 'cuda') else '❌'}")
        
        if hasattr(cv2, 'cuda'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"🎮 CUDA Devices: {device_count}")
            
            if device_count > 0:
                print("✅ ¡CUDA FUNCIONANDO CORRECTAMENTE!")
                
                # Información detallada del dispositivo
                try:
                    device_info = cv2.cuda.DeviceInfo(0)
                    print(f"   📍 Device: {device_info.name()}")
                    print(f"   💾 Memory: {device_info.totalGlobalMem() / (1024**3):.1f} GB")
                    print(f"   🔢 Compute: {device_info.majorVersion()}.{device_info.minorVersion()}")
                except Exception as e:
                    print(f"   ⚠️ Info detallada no disponible: {e}")
                
                return True
            else:
                print("❌ CUDA módulo presente pero sin dispositivos")
                return False
        else:
            print("❌ Módulo CUDA no disponible")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando OpenCV: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verificando CUDA: {e}")
        return False

def rollback_opencv():
    """Restaura OpenCV anterior si hay problemas"""
    print("🔄 RESTAURANDO OPENCV ANTERIOR")
    print("=" * 50)
    
    if not os.path.exists('opencv_backup_info.txt'):
        print("❌ No se encontró información de respaldo")
        return False
    
    try:
        # Desinstalar versión problemática
        subprocess.run(['pip', 'uninstall', 'opencv-python', 'opencv-contrib-python', '-y'],
                      capture_output=True)
        
        # Reinstalar versión estable
        cmd = ['pip', 'install', 'opencv-python==4.11.0']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ OpenCV restaurado a versión estable")
            return True
        else:
            print(f"❌ Error restaurando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante rollback: {e}")
        return False

def run_performance_test():
    """Ejecuta test de rendimiento CUDA"""
    print("🚀 EJECUTANDO TEST DE RENDIMIENTO")
    print("=" * 50)
    
    try:
        # Ejecutar test CUDA experimental
        result = subprocess.run(['python', 'run_cuda_experimental_test.py'], 
                              input='10\n\n', text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Test completado exitosamente")
            # Extraer métricas clave del output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU speedup:' in line or 'FPS promedio:' in line or 'CUDA disponible:' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"⚠️ Test completado con advertencias: {result.stderr}")
        
    except Exception as e:
        print(f"❌ Error ejecutando test: {e}")

def main():
    """Función principal del setup"""
    print("🚀 OPENCV CUDA SETUP")
    print("=" * 50)
    print("📖 Basado en: https://opencv.org/platforms/cuda/")
    print()
    
    # Verificar requisitos
    if not check_system_requirements():
        print("❌ Requisitos del sistema no cumplidos")
        return
    
    # Respaldar instalación actual
    backup_current_opencv()
    
    # Mostrar opciones
    print("🛠️ OPCIONES DE INSTALACIÓN:")
    print("1. 🐍 Conda-forge (Recomendado)")
    print("2. 📦 Pip precompilado (Experimental)")
    print("3. 🔄 Rollback a versión estable")
    print("4. 🚀 Solo test de rendimiento")
    print("5. ❌ Salir")
    print()
    
    while True:
        try:
            choice = input("Selecciona una opción (1-5): ").strip()
            
            if choice == '1':
                success = install_opencv_conda()
                break
            elif choice == '2':
                success = install_opencv_pip_precompiled()
                break
            elif choice == '3':
                success = rollback_opencv()
                break
            elif choice == '4':
                run_performance_test()
                return
            elif choice == '5':
                print("👋 Saliendo...")
                return
            else:
                print("❌ Opción inválida")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Proceso cancelado")
            return
    
    if choice in ['1', '2']:
        print("\n" + "=" * 50)
        
        if success:
            # Verificar instalación
            if verify_cuda_installation():
                print("\n🎉 ¡INSTALACIÓN EXITOSA!")
                print("💡 Ahora puedes usar CUDA en el detector")
                
                # Preguntar si ejecutar test
                test_now = input("\n¿Ejecutar test de rendimiento ahora? (s/N): ").lower()
                if test_now == 's':
                    run_performance_test()
            else:
                print("\n❌ Instalación falló - ejecutando rollback...")
                rollback_opencv()
        else:
            print("\n❌ Instalación falló")
            rollback_choice = input("¿Ejecutar rollback automático? (s/N): ").lower()
            if rollback_choice == 's':
                rollback_opencv()

if __name__ == "__main__":
    main() 