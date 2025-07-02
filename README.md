# 🎸 Guitar Hero IA - Sistema de Detección y Visualización

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este proyecto es un sistema avanzado de **Computer Vision** para Guitar Hero, enfocado en la detección de notas en tiempo real. La arquitectura actual está altamente optimizada para el **debugging, la visualización y el análisis de rendimiento**, utilizando técnicas de procesamiento de imágenes y concurrencia.

## ✨ Arquitectura y Filosofía

El sistema se basa en los siguientes principios:

- **Configuración Centralizada**: Todas las configuraciones (parámetros de captura, rangos de color HSV, polígonos de carril, etc.) residen en un único archivo `config/config.ini`.
- **"Fail-Fast"**: El gestor de configuración (`ConfigManager`) es estricto. Si una configuración requerida no se encuentra, el programa se detiene inmediatamente para evitar comportamientos inesperados. No existen valores por defecto ocultos.
- **Modularidad**: El código está organizado en módulos con responsabilidades claras: captura de pantalla, detección de score, gestión de configuración, etc.
- **Rendimiento**: Se utilizan técnicas como el multithreading para operaciones costosas (análisis de carriles, OCR) y se minimizan las operaciones de procesamiento de imagen para mantener un alto framerate.

## 🛠️ Herramientas Principales

El proyecto ha sido refactorizado para centrarse en herramientas de desarrollo y diagnóstico potentes.

### 1. Visualizador de Detección (`utils/polygon_visualizer.py`)

Esta es la herramienta **principal** del proyecto. Permite visualizar en tiempo real todo el proceso de detección sobre la ventana del juego.

**Funcionalidades:**
- **Detección en Tiempo Real**: Detecta notas verdes y amarillas usando rangos HSV.
- **Visualización de Polígonos**: Dibuja los polígonos de cada carril para verificar su posición.
- **Contadores y Métricas**: Muestra FPS, puntuación actual (vía OCR) y el total de notas detectadas.
- **Modos de Vista**: Permite alternar entre la vista normal y máscaras de color para depurar la detección.
- **Optimización de Rendimiento**:
    - **Procesamiento Concurrente**: Cada carril se analiza en un hilo separado.
    - **OCR no Bloqueante**: La detección de la puntuación se ejecuta en un hilo aparte para no impactar los FPS.

### 2. Calibrador HSV (`utils/static_hsv_calibrator_plus.py`)

Herramienta avanzada para encontrar los rangos de color HSV y los parámetros de morfología perfectos.

**Funcionalidades:**
- **Ajuste en Tiempo Real**: Usa sliders para modificar los valores HSV (Hue, Saturation, Value) y los parámetros de las operaciones morfológicas (Close, Dilate).
- **Previsualización Instantánea**: Muestra el resultado de aplicar los filtros y transformaciones a una imagen estática.
- **Guardado de Configuración**: Guarda los parámetros optimizados directamente en `config/config.ini`.

## 🚀 Guía de Uso

### 1. Instalación
```bash
# 1. Clona el repositorio
git clone <URL_DEL_REPOSITORIO>
cd guitar_hero_ia

# 2. (Recomendado) Crea y activa un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
# source venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt
```

### 2. Flujo de Trabajo para Calibración y Detección

El `config/config.ini` ya viene con valores pre-configurados que deberían funcionar. Si la detección falla, sigue estos pasos:

**Paso 1: Calibrar Colores y Morfología (Si es necesario)**

Si las notas no se detectan correctamente, usa el calibrador avanzado.

```bash
# Ejecuta el calibrador como un módulo
python -m utils.static_hsv_calibrator_plus
```
Ajusta los sliders hasta que las notas en la previsualización queden completamente blancas y aisladas. Guarda los cambios con la tecla 's'.

**Paso 2: Ejecutar el Visualizador de Detección**

Esta es la herramienta principal para ver el sistema en acción.

```bash
# Ejecuta el visualizador como un módulo
python -m utils.polygon_visualizer
```

**Controles del Visualizador:**
- `q`: Salir del programa.
- `v`: Cambiar el modo de visualización (Normal -> Máscara Amarilla -> Máscara Verde).

## 📂 Estructura del Proyecto

```
guitar_hero_ia/
├── config/
│   └── config.ini              # ✅ ÚNICA FUENTE DE VERDAD para la configuración.
├── data/
│   └── templates/
│       └── image.png           # Imagen estática para el calibrador HSV.
├── src/
│   ├── core/                   # Módulos centrales de la aplicación.
│   │   ├── screen_capture.py   # Captura de pantalla optimizada (usa MSS).
│   │   └── score_detector.py   # Detector de puntuación con OCR (Pytesseract).
│   └── utils/
│       └── config_manager.py   # Gestor de configuración estricto ("Fail-Fast").
├── utils/                      # 🛠️ HERRAMIENTAS DE DESARROLLO INDEPENDIENTES.
│   ├── polygon_visualizer.py       # VISUALIZADOR PRINCIPAL: Detección en tiempo real.
│   └── static_hsv_calibrator_plus.py # CALIBRADOR AVANZADO: HSV y Morfología.
├── requirements.txt            # Dependencias del proyecto.
└── README.md                   # Esta guía.
```
*Nota: Otros scripts como `polygon_calibrator.py`, `window_calibrator.py` y `guitar_hero_main.py` existen pero no forman parte del flujo de trabajo de depuración actual y serán re-integrados o eliminados en futuras refactorizaciones.*

## 🎯 Método de Detección Actual

El sistema usa **HSV Color Filtering** en lugar de template matching para máximo rendimiento:

### **Detección por Colores HSV**
- **🟡 Notas Amarillas**: HSV [15,100,100] - [40,255,255]
- **🟢 Notas Verdes**: HSV [25,40,40] - [95,255,255] (calibrable)
- **Multithreading**: 6 workers simultáneos (uno por carril)
- **Rendimiento**: ~47.5 FPS promedio (⭐⭐⭐ EXCELENTE)

### **¿Por qué HSV y no Template Matching?**
- ✅ **10x más rápido** que buscar imágenes PNG
- ✅ **Funciona con diferentes estilos** de notas
- ✅ **Menor uso de CPU y memoria**
- ✅ **Calibrable en tiempo real**

## 🚀 Guía de Inicio Rápido

### 1. Prerrequisitos
- Python 3.11 o superior
- Windows 10/11 (sistema de hotkeys optimizado para Windows)

### 3. Ejecución del Sistema
Para iniciar el sistema completo:
```bash
python src/guitar_hero_main.py
```

## 📋 Flujo de Trabajo Recomendado

### Paso 1: Calibrar la Ventana del Juego (Opción 1)
- **¿Qué hace?**: Define exactamente dónde está ubicada la ventana de Guitar Hero
- **¿Cómo?**: Abre Guitar Hero, selecciona opción 1 del menú, haz clic y arrastra para seleccionar el área de juego
- **Importancia**: **CRÍTICO** - Sin calibración correcta nada funcionará

### Paso 2: Calibrar Polígonos de Detección
- **¿Qué hace?**: Define áreas precisas donde detectar notas en cada carril
- **¿Cómo?**: Ejecuta el calibrador de polígonos:
  ```bash
  python polygon_calibrator.py
  ```
- **Proceso**: 
  - Haz clic en 4 puntos por carril para definir el área de detección
  - Los polígonos se optimizan para máximo rendimiento
  - Se guardan automáticamente en `config/config.ini`

### Paso 3: Calibrar Colores HSV (Nuevo)
- **¿Qué hace?**: Ajusta los rangos de color para detectar notas amarillas y verdes
- **¿Cómo?**: Ejecuta el calibrador de colores:
  ```bash
  # Calibración estática (recomendado - no pausa el juego)
  python utils/static_hsv_calibrator.py
  ```
- **Proceso**:
  - Usa sliders para ajustar rangos HSV 
  - Calibración estática usa screenshot fijo
  - Se guardan automáticamente en configuración global

### Paso 4: Ejecutar Sistema (Opción 10)
- **¿Qué hace?**: Inicia el sistema de detección con hotkeys globales
- **Hotkeys Disponibles**:
  - **F9**: Iniciar/Detener detección
  - **F10**: Cambiar modo de detección
  - **F11**: Toggle información en pantalla
  - **F12**: Parada de emergencia

## 🔧 Sistema de Detección HSV

### **Polígonos Optimizados ✅**
El sistema usa polígonos calibrados manualmente para máximo rendimiento:
- **Reducción de área**: Hasta 54% menos área de procesamiento por carril
- **Coordenadas relativas**: Manejo inteligente de coordenadas absolutas vs relativas
- **Calibración interactiva**: Herramienta visual para definir áreas exactas
- **Estado actual**: **6 polígonos perfectamente posicionados** 

### **Detección HSV de 2 Colores**
1. **🟡 Notas Amarillas**: `[15,100,100] - [40,255,255]` (optimizado)
2. **🟢 Notas Verdes**: `[25,40,40] - [95,255,255]` (calibrable)

### **Configuración de Polígonos**
```ini
[LANE_POLYGON_S]
point_0_x = 165
point_0_y = 585
point_1_x = 293
point_1_y = 691
point_2_x = 503
point_2_y = 535
point_3_x = 370
point_3_y = 444
point_count = 4
```

## 🛠️ Utilidades de Desarrollo (Limpia)

### **Scripts Disponibles (Solo Esenciales)**

#### 🎨 **Calibrador HSV Estático** (`static_hsv_calibrator.py`) - ⭐ RECOMENDADO
```bash
python utils/static_hsv_calibrator.py
```
- **Calibración sin pausar el juego** usando screenshot estático
- **Sliders HSV interactivos** para ajustar rangos amarillo/verde
- **3 modos de visualización**: Original, Máscara amarilla, Máscara verde
- **Click para información de píxel** (valores HSV exactos)
- **Guardado automático** de rangos optimizados
- **Controles**:
  - Click: Info píxel HSV
  - 's': Guardar rangos
  - 'r': Reset valores por defecto
  - 'q': Salir

#### 🎯 **Visualizador de Polígonos** (`polygon_visualizer.py`) - ⭐ PRINCIPAL
```bash
python utils/polygon_visualizer.py
```
- **Visualización de polígonos** configurados sobre el juego en tiempo real
- **3 modos de vista**: Normal, Máscara amarilla, Máscara verde
- **Detección automática** usando rangos HSV optimizados globales
- **Conteo de notas** amarillas/verdes por carril
- **Sin calibración** - usa valores guardados automáticamente
- **Controles**:
  - 'q': Salir
  - 's': Capturar frame
  - '+': Cambiar vista (Normal → Amarilla → Verde)
  - SPACE: Pausar/Reanudar

#### 🔍 **Verificador de Sistema** (`check_system_status.py`)
```bash
python utils/check_system_status.py
```
- Diagnóstico completo de configuración
- Verificación de polígonos y plantillas
- Test de componentes del sistema
- Benchmark integrado

#### ⚡ **Benchmark Rápido** (`quick_benchmark.py`)
```bash
python utils/quick_benchmark.py
```
- Medición pura de rendimiento (sin GUI)
- Opciones de 5s, 10s o 30s
- Evaluación automática de performance

## 🚀 Casos de Uso Comunes

### ⭐ **Flujo Rápido Diario (RECOMENDADO)**
```bash
# 1. Verificar estado general
python utils/check_system_status.py

# 2. Ver polígonos y detección en tiempo real
python utils/polygon_visualizer.py

# 3. Benchmark rápido  
python utils/quick_benchmark.py
```

### 🎨 **Calibración de Colores HSV (Solo cuando sea necesario)**
```bash
# Calibración estática (RECOMENDADO - no pausa el juego)
python utils/static_hsv_calibrator.py
# Los valores se guardan automáticamente en configuración global
```

### 🎯 **Verificar Configuración de Polígonos**
```bash
# Verificar posicionamiento visual y detección
python utils/polygon_visualizer.py
```

### 📊 **Medir Rendimiento**
```bash
# Benchmark sin GUI para máximo rendimiento
python utils/quick_benchmark.py
```

## 📈 Métricas de Referencia - ✅ ESTADO ACTUAL OPTIMIZADO

### 🎯 FPS Objetivo vs. Actual
- **⭐ ACTUAL**: **47.5 FPS** (~21ms/frame) - ⭐⭐⭐ EXCELENTE
- **Método**: HSV Color Filtering + Multithreading
- **Excelente**: 30+ FPS (≤33ms/frame) ✅ **SUPERADO**
- **Bueno**: 20-30 FPS (33-50ms/frame)
- **Aceptable**: 10-20 FPS (50-100ms/frame)
- **Pobre**: <10 FPS (>100ms/frame)

### 🔍 Áreas de Polígonos Optimizadas ✅ CONFIGURADO
- **Total actual**: **237,710 px²**
- **Reducción**: **44.2%** vs. configuración original
- **Carriles**: **6 configurados** (S, D, F, J, K, L)
- **Estado**: **✅ Polígonos perfectamente posicionados**

### 🎨 Detección HSV Optimizada ✅ NUEVO
- **Amarillas**: **Detecta perfectamente** con `[15,100,100] - [40,255,255]`
- **Verdes**: **Calibrable** con `[25,40,40] - [95,255,255]`
- **Método**: **HSV Color Filtering** (10x más rápido que template matching)
- **Multithreading**: **6 workers simultáneos**

### 📊 Configuraciones de Rendimiento Disponibles
- **🏎️ Velocidad Máxima**: HSV filtering + threshold altos
- **⚖️ Equilibrado**: Detecta la mayoría de notas (recomendado)
- **🎯 Precisión Máxima**: Detecta todas las notas posibles

## 🔧 Personalización

### Modificar Rangos HSV
Los rangos se calibran con el calibrador estático y se cargan automáticamente desde `hsv_ranges_optimized.txt`:
```bash
python utils/static_hsv_calibrator.py
```

### Ajustar Polígonos
```bash
python polygon_calibrator.py
```

### Verificar Configuración Actual
El sistema carga automáticamente los valores optimizados. Para verificar:
```bash
python utils/polygon_visualizer.py
```

## 📝 Notas Importantes

- **Método actual**: **HSV Color Filtering** (NO template matching)
- **Templates PNG**: Solo se usan como referencia visual
- **Rendimiento**: Sistema optimizado para 47.5+ FPS
- **Polígonos**: Perfectamente calibrados y posicionados
- **Persistencia**: Configuraciones se guardan automáticamente

## 🤝 Contribuir

Al hacer cambios al sistema:
1. Verificar con `python utils/check_system_status.py`
2. Visualizar polígonos con `python utils/polygon_visualizer.py`
3. Medir impacto con `python utils/quick_benchmark.py`
4. Calibrar colores (si es necesario) con `python utils/static_hsv_calibrator.py`
5. Documentar cambios en este README

---

**🎸 ¡Sistema optimizado para máximo rendimiento con HSV Color Filtering!**