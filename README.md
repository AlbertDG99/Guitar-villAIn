# 🎸 Guitar Hero IA - Sistema de Detección Optimizado 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema avanzado de detección de notas para Guitar Hero con Computer Vision optimizado y HSV Color Filtering. El proyecto incluye calibración avanzada, detección por colores con polígonos optimizados y sistema de hotkeys global.

## 🌟 Características Principales

- **Detección HSV Optimizada**: Sistema de detección por colores HSV con polígonos calibrados para máximo rendimiento
- **Calibración Avanzada**: Herramientas para calibrar ventana del juego, polígonos y rangos de color HSV
- **Sistema de Hotkeys**: Control global que funciona sin importar la ventana activa
- **Detector Híbrido**: HSV Color Filtering + Multithreading (6 workers) para detección súper rápida
- **Configuración Inteligente**: Sistema que maneja coordenadas absolutas y relativas automáticamente

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

### 2. Instalación
```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/guitar_hero_ia.git
cd guitar_hero_ia

# 2. Crea y activa un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate

# 3. Instala las dependencias
pip install -r requirements.txt
```

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

## 📂 Estructura del Proyecto (Optimizada)

```
guitar_hero_ia/
├── config/
│   └── config.ini              # Configuración principal (regiones, colores, polígonos)
├── data/
│   ├── metrics.json            # Métricas del sistema
│   └── templates/              # Plantillas y screenshots para calibración
│       ├── yellow_star.png     # Plantilla nota amarilla (referencia)
│       ├── green_star_start.png # Plantilla inicio nota verde (referencia)
│       ├── green_star_end.png  # Plantilla fin nota verde (referencia)
│       └── image.png           # Screenshot para calibración HSV
├── src/
│   ├── core/                   # Módulos centrales
│   │   ├── screen_capture.py   # Captura de pantalla optimizada
│   │   ├── note_detector.py    # Detector principal de notas
│   │   ├── input_controller.py # Control de entrada (teclado)
│   │   ├── timing_system.py    # Sistema de timing
│   │   └── score_detector.py   # Detector de puntuación
│   ├── utils/                  # Utilidades del sistema
│   │   ├── config_manager.py   # Gestor de configuración
│   │   ├── logger.py          # Sistema de logs
│   │   └── overlay.py         # Overlay visual
│   ├── ai/                     # Sistema de IA (en desarrollo)
│   │   └── dqn_agent.py       # Agente DQN (deshabilitado)
│   ├── guitar_hero_main.py     # **PUNTO DE ENTRADA PRINCIPAL**
│   ├── guitar_hero_hotkeys.py  # Sistema de hotkeys
│   ├── window_calibrator.py    # Calibrador de ventana
│   ├── note_line_calibrator.py # Calibrador de líneas de notas
│   ├── hotkey_controller.py    # Controlador de hotkeys
│   └── monitor_setup.py        # Configuración de monitores
├── utils/                      # **🛠️ UTILIDADES DE DESARROLLO (LIMPIA)**
│   ├── polygon_visualizer.py          # **VISUALIZADOR DE POLÍGONOS Y DETECCIÓN**
│   ├── static_hsv_calibrator.py       # **CALIBRADOR HSV ESTÁTICO (SIN PAUSAR JUEGO)**
│   ├── check_system_status.py         # Verificador de estado del sistema
│   └── quick_benchmark.py             # Benchmark rápido de FPS
├── polygon_calibrator.py       # **CALIBRADOR DE POLÍGONOS**
├── requirements.txt            # Dependencias
└── README.md                  # Esta guía
```

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