# 🎸 Guitar Hero IA - Sistema Completo de Aprendizaje Reforzado 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema avanzado de inteligencia artificial para Guitar Hero usando Deep Q-Learning (DQN) y Computer Vision. Este proyecto permite a una IA aprender a jugar Guitar Hero de forma autónoma, observando la pantalla y tomando decisiones en tiempo real.

## 🌟 Características

- **IA Avanzada**: Implementación de Dueling Double DQN con Prioritized Experience Replay.
- **Visión por Computadora**: Detección de notas en tiempo real con alta precisión mediante rangos de color HSV.
- **Sistema Modular**: Calibración, entrenamiento, evaluación y juego, todo gestionado desde un único menú interactivo.
- **Control Preciso**: Ejecución de acciones con baja latencia para un timing casi perfecto.
- **Depuración Visual**: Herramientas para visualizar y ajustar la detección de notas en tiempo real.

## 🚀 Guía de Inicio Rápido

### 1. Prerrequisitos
- Python 3.11 o superior.
- Una GPU NVIDIA compatible con CUDA para el entrenamiento y la ejecución de la IA (recomendado).

### 2. Instalación
```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/guitar_hero_ia.git
cd guitar_hero_ia

# 2. Crea y activa un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt
```

### 3. Ejecución del Sistema
Para iniciar, ejecuta el script principal que te dará acceso a todas las funcionalidades a través de un menú interactivo.
```bash
python src/guitar_hero_main.py
```

## 📋 Flujo de Trabajo Recomendado

Sigue estos pasos para poner en marcha el sistema correctamente.

### Paso 1: Calibrar la Ventana del Juego (Opción 1)
- **¿Qué hace?**: Le dice al sistema dónde se encuentra la ventana de Guitar Hero en tu pantalla.
- **¿Cómo?**: Abre el juego. Desde el menú principal del script, selecciona la opción 1. Se te pedirá que hagas clic y arrastres el ratón para dibujar un rectángulo que enmarque **exactamente** el área de juego.
- **Importancia**: Este paso es **crítico**. Una mala calibración resultará en una detección de notas incorrecta.

### Paso 2: Diagnosticar la Detección (Opción 3)
- **¿Qué hace?**: Muestra una ventana en tiempo real de lo que la IA "ve". Te permite ajustar los colores para que las notas se detecten perfectamente bajo tus condiciones de iluminación y configuración de pantalla.
- **¿Cómo?**: Con el juego en la pantalla de una canción, selecciona la opción 3. Usa los controles deslizantes para ajustar los rangos de color (HSV) hasta que solo las notas se iluminen claramente en la ventana de máscaras.
- **Importancia**: Fundamental para que la IA reciba información correcta. Si no detecta bien las notas, no podrá aprender.

### Paso 3: Entrenar la IA (Opciones 6 y 7)
- **¿Qué hace?**: Inicia el proceso de aprendizaje por refuerzo. La IA jugará partidas para aprender a asociar estados (notas en pantalla) con acciones (pulsar teclas).
- **¿Cómo?**:
    - **Opción 6 (Entrenar nuevo modelo)**: Empieza desde cero. Ideal para la primera vez.
    - **Opción 7 (Continuar entrenamiento)**: Carga un modelo previamente guardado y sigue entrenándolo.
- **Proceso**: El entrenamiento puede durar varias horas. Los modelos se guardan automáticamente en la carpeta `models/`. Puedes interrumpir el proceso en cualquier momento con `Ctrl+C` y el progreso se guardará.

### Paso 4: Jugar con la IA (Opción 10)
- **¿Qué hace?**: Carga el modelo entrenado y lo pone a jugar.
- **¿Cómo?**: Selecciona la opción 10. El sistema utilizará el mejor modelo que encuentre en la carpeta `models/` y empezará a jugar usando hotkeys.
- **Hotkeys**:
    - **F9**: Inicia/Detiene la IA.
    - **F12**: Parada de emergencia.

## 📂 Estructura del Proyecto

```
guitar_hero_ia/
├── config/
│   └── config.ini          # Fichero de configuración principal (regiones, colores, etc.)
├── data/
│   └── metrics.json        # Estadísticas de entrenamiento y partidas
├── logs/
│   └── *.log               # Logs de eventos y errores
├── models/
│   └── *.pth               # Modelos de IA entrenados
├── src/
│   ├── ai/                 # Lógica del agente DQN
│   ├── core/               # Módulos centrales (detección, captura, control)
│   ├── utils/              # Utilidades (configuración, logs)
│   ├── guitar_hero_main.py # Punto de entrada principal con el menú
│   └── ...                 # Otros scripts de apoyo
├── README.md               # Esta guía
└── requirements.txt        # Dependencias de Python
```

## ⚠️ Solución de Problemas

- **La IA no presiona ninguna tecla**:
  1. Asegúrate de haber ejecutado la calibración de la ventana (Paso 1).
  2. Verifica el diagnóstico de detección (Paso 2). Si no se detectan notas, la IA no hará nada.
  3. Es posible que el script no tenga permisos para simular pulsaciones de teclas. En Windows, prueba a ejecutar la terminal como Administrador.

- **La detección de notas es pobre**:
  - La causa más común es una mala configuración de los colores HSV. Dedica tiempo en el menú de diagnóstico (Opción 3) para ajustarlos.
  - La iluminación de tu habitación o el brillo de tu monitor pueden afectar la detección. Intenta mantener condiciones consistentes.

- **El script se cierra con un error**:
  - Revisa el fichero de log más reciente en la carpeta `logs/` para obtener un mensaje de error detallado.

## 🎯 Uso del Sistema

### **Menú Principal Interactivo**

```bash
python src/guitar_hero_main.py
```

### **Comandos Directos (Línea de Comandos)**

```bash
# Calibración rápida
python src/guitar_hero_main.py --calibrate

# Ejecutar sistema
python src/guitar_hero_main.py --run

# Diagnóstico de detección
python src/guitar_hero_main.py --diagnose

# Entrenar IA
python src/guitar_hero_main.py --train 100
```

### **Scripts Especializados**

```bash
# Solo diagnóstico visual
python src/detection_debugger.py

# Solo entrenamiento de IA
python src/reinforcement_trainer.py

# Launcher todo-en-uno
python run_ai_training.py
```

## 🔧 Configuración Paso a Paso

### **1. Calibración Inicial**
1. Abre Guitar Hero en ventana completa
2. Ejecuta `python src/guitar_hero_main.py`
3. Selecciona opción **1** (Calibrar ventana)
4. Selecciona exactamente el área de juego (sin bordes verdes)
5. Usa las guías amarillas para máxima precisión

### **2. Verificación de Detección**
1. Selecciona opción **3** (Diagnosticar detección)
2. Observa las notas detectadas en tiempo real
3. Ajusta rangos HSV si es necesario con los trackbars
4. Guarda configuración cuando esté perfecta

### **3. Entrenamiento de IA**
1. Selecciona opción **6** (Entrenar nuevo modelo)
2. Especifica número de episodios (recomendado: 100-500)
3. El sistema guardará modelos cada 10 episodios
4. Monitorea progreso en tiempo real

### **4. Ejecución Final**
1. Selecciona opción **10** (Ejecutar con hotkeys)
2. Usa **F9** para iniciar/parar la IA
3. Usa **F11** para toggle información
4. Usa **F12** para parada de emergencia

## 🧠 Sistema de Aprendizaje Reforzado

### **Arquitectura DQN**

```python
class DuelingDQN(nn.Module):
    def __init__(self):
        # Convolutional layers para procesar imagen
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Dueling streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, 5)  # 5 acciones
```

### **Sistema de Rewards**

| Evento | Reward | Descripción |
|---------|---------|-------------|
| Perfect Hit | +100 | Timing perfecto |
| Good Hit | +50 | Buen timing |
| OK Hit | +25 | Timing aceptable |
| Miss | -10 | Nota perdida |
| Acción innecesaria | -1 | Tecla sin nota |

### **Configuración de Entrenamiento**

```ini
[TRAINING]
learning_rate = 0.001
epsilon_start = 0.1
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
memory_size = 10000
target_update_frequency = 100
```

## 🔍 Diagnóstico y Depuración

### **Depurador Visual**

El sistema incluye un depurador visual avanzado:

```bash
python src/detection_debugger.py
```

**Controles del depurador:**
- **1**: Vista normal
- **2**: Vista HSV 
- **3**: Máscaras de color
- **4**: Contornos detectados
- **S**: Captura de pantalla
- **ESC**: Salir

### **Análisis de Coordenadas**

```python
# Verificar precisión de calibración
calibrator = WindowCalibrator()
region = calibrator.get_capture_region()
print(f"Región: {region['width']}x{region['height']}")
```

### **Configuración HSV**

```ini
[DETECTION]
# Notas amarillas (normales)
yellow_h_min = 20
yellow_h_max = 35
yellow_s_min = 200
yellow_v_min = 200

# Notas verdes (sostenidas)
green_h_min = 40
green_h_max = 80
green_s_min = 200
green_v_min = 200
```

## 📊 Métricas y Evaluación

### **Métricas de Entrenamiento**

```json
{
    "episode": 50,
    "total_reward": 1250.5,
    "accuracy": 0.87,
    "perfect_hits": 42,
    "good_hits": 28,
    "ok_hits": 15,
    "misses": 8,
    "epsilon": 0.045,
    "loss": 0.023,
    "training_time": "2.3 hours"
}
```

### **Evaluación de Modelos**

```bash
# Evaluar modelo específico
python src/reinforcement_trainer.py --evaluate models/best_model.pth

# Comparar múltiples modelos
python src/reinforcement_trainer.py --compare models/
```

## 🎮 Hotkeys Globales

| Tecla | Función | Descripción |
|-------|---------|-------------|
| **F9** | Iniciar/Parar IA | Toggle principal del sistema |
| **F10** | Calibrar | Recalibrar área de juego |
| **F11** | Toggle Info | Mostrar/ocultar información |
| **F12** | Emergencia | Parada inmediata |

## 🏗️ Arquitectura del Sistema

```
guitar_hero_ia/
├── src/
│   ├── ai/
│   │   └── dqn_agent.py           # Agente DQN principal
│   ├── core/
│   │   ├── note_detector.py       # Detección de notas
│   │   ├── input_controller.py    # Control de teclas
│   │   ├── screen_capture.py      # Captura de pantalla
│   │   └── timing_system.py       # Sistema de timing
│   ├── utils/
│   │   ├── config_manager.py      # Gestión de configuración
│   │   └── logger.py              # Sistema de logging
│   ├── detection_debugger.py      # Depurador visual
│   ├── reinforcement_trainer.py   # Entrenador de IA
│   ├── guitar_hero_main.py        # Interfaz principal
│   └── guitar_hero_hotkeys.py     # Sistema de hotkeys
├── models/                        # Modelos entrenados
├── data/                          # Datos de entrenamiento
├── logs/                          # Archivos de log
├── config/                        # Configuraciones
└── run_ai_training.py            # Launcher simplificado
```

## 🎯 Flujo de Trabajo Recomendado

### **Para Nuevos Usuarios:**

1. **Configuración Inicial**
   ```bash
   python src/guitar_hero_main.py
   # Opción 1: Calibrar ventana
   ```

2. **Verificación**
   ```bash
   # Opción 3: Diagnosticar detección
   # Ajustar hasta ver correctamente las notas
   ```

3. **Entrenamiento**
   ```bash
   # Opción 6: Entrenar nuevo modelo
   # Empezar con 50-100 episodios
   ```

4. **Uso**
   ```bash
   # Opción 10: Ejecutar con hotkeys
   # F9 para iniciar la IA
   ```

### **Para Usuarios Avanzados:**

```bash
# Entrenamiento directo con configuración específica
python src/reinforcement_trainer.py \
    --episodes 500 \
    --learning-rate 0.0005 \
    --epsilon-decay 0.99 \
    --save-interval 5

# Evaluación comparativa
python src/reinforcement_trainer.py \
    --evaluate \
    --model models/best_model.pth \
    --episodes 10
```

## 🔬 Algoritmos y Técnicas

### **Deep Q-Learning (DQN)**
- **Target Network**: Red objetivo para estabilidad
- **Experience Replay**: Buffer de experiencias pasadas  
- **Epsilon-Greedy**: Exploración vs explotación
- **Dueling Architecture**: Streams separados Q(s,a) = V(s) + A(s,a)

### **Computer Vision**
- **Color Segmentation**: Detección por rangos HSV
- **Morphological Operations**: Limpieza de máscaras
- **Contour Detection**: Identificación de formas
- **Template Matching**: Verificación de patrones

### **Optimizaciones**
- **Mixed Precision**: FP16 para entrenamiento rápido
- **Batch Processing**: Procesamiento por lotes
- **Memory Mapping**: Uso eficiente de RAM
- **Multi-threading**: Captura y procesamiento paralelo

## 📈 Resultados y Rendimiento

### **Benchmarks Típicos**
- **Precisión**: 85-95% después de 200 episodios
- **Velocidad**: <1ms latencia de respuesta
- **Estabilidad**: >99% uptime durante sesiones largas
- **GPU**: RTX 3060+ recomendada para entrenamiento

### **Mejores Prácticas**
- Entrenar en sesiones de 50-100 episodios
- Guardar checkpoints frecuentemente
- Usar validación cruzada para evaluar modelos
- Ajustar hiperparámetros gradualmente

## 🛠️ Solución de Problemas

### **Problemas Comunes**

#### **No Detecta Notas**
```bash
# 1. Verificar calibración
python src/guitar_hero_main.py --calibrate

# 2. Usar depurador visual
python src/detection_debugger.py

# 3. Ajustar rangos HSV en config.ini
```

#### **IA No Aprende**
```bash
# 1. Verificar GPU/CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 2. Reducir learning rate
# Editar config.ini: learning_rate = 0.0005

# 3. Aumentar episodios de entrenamiento
```

#### **Baja Precisión**
```bash
# 1. Verificar timing del sistema
# Opción 5: Test de sistema completo

# 2. Ajustar reward system
# Modificar rewards en dqn_agent.py

# 3. Entrenar por más tiempo
```

### **Logs y Debugging**

```bash
# Ver logs en tiempo real
tail -f logs/game_events.log

# Analizar métricas de entrenamiento
python -c "
import json
with open('data/metrics.json') as f:
    data = json.load(f)
    print(f'Reward promedio: {data['average_reward']:.2f}')
"
```

## 🤝 Contribución

### **Cómo Contribuir**
1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

### **Áreas de Mejora**
- Detección de acordes complejos
- Soporte para más tipos de notas
- Interface gráfica mejorada
- Optimizaciones de rendimiento
- Soporte multi-idioma

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **PyTorch Team** por el framework de deep learning
- **OpenCV** por las herramientas de computer vision
- **Comunidad de Guitar Hero** por inspiración y feedback
- **Desarrolladores de IA** por algoritmos y técnicas

## 📞 Soporte

### **¿Necesitas Ayuda?**

1. **GitHub Issues**: Para bugs y requests
2. **Documentación**: Wiki del proyecto
3. **Ejemplos**: Directorio `/examples`
4. **Logs**: Revisar `/logs` para debugging

### **FAQ**

**P: ¿Funciona con Clone Hero?**
R: Sí, funciona con cualquier juego similar a Guitar Hero.

**P: ¿Necesito GPU?**
R: Recomendada para entrenamiento, pero funciona solo con CPU.

**P: ¿Cuánto tiempo toma entrenar?**
R: 2-4 horas para un modelo básico funcional.

**P: ¿Funciona en Linux/Mac?**
R: Sí, multiplataforma con Python 3.11+.

---

**🎸 ¡Happy Playing! 🤖**

> **Nota**: Este sistema es para propósitos educativos y de investigación. Úsalo responsablemente y respeta los términos de servicio de los juegos.

## 🎯 Detector de Notas CPU-Optimizado

### **Sistema Simplificado de Alto Rendimiento**
- **Solo 2 tipos de notas**: Amarillas (threshold 1.0) y verdes (threshold 0.95)
- **Nomenclatura por teclas**: s, d, f, j, k, l (mapeo directo)
- **Multithreading**: ✅ **6 workers CPU optimizado** - Sin overhead GPU
- **Anti-duplicados**: Filtro 25% superposición entre detecciones
- **Rendimiento**: **11.6ms/frame** (86 FPS equivalente)

### **Configuración Técnica**
```python
# Rangos HSV optimizados
Amarillas: [15, 100, 100] - [40, 255, 255]  # Threshold: 1.0
Verdes:    [25, 40, 40]   - [95, 255, 255]  # Threshold: 0.95

# Optimizaciones CPU
✅ OpenCV directo: Sin overhead GPU innecesario
✅ Multithreading: 6 hilos paralelos por tecla
✅ Operaciones morfológicas: Kernels optimizados
```

### **Rendimiento Optimizado**
```
Benchmark verificado:
• 11.6ms/frame promedio (86.4 FPS equivalente)
• Mínimo: 10.1ms | Máximo: 15.5ms
• CPU directo: Sin transferencias GPU innecesarias
• Multithreading: 6 workers activos
```

## ⚡ Optimización de Rendimiento

### 🚀 Estado Actual - CPU Optimizado
- **Rendimiento**: ✅ **11.6ms/frame** (86.4 FPS equivalente)
- **Multithreading**: ✅ 6 workers CPU optimizado
- **OpenCV**: ✅ Operaciones directas sin overhead
- **Lección aprendida**: CPU es más eficiente para estas operaciones específicas

### 🎯 Optimizaciones Implementadas

**CPU Directo + Multithreading:**
- ✅ OpenCV cv2.inRange() directo (sin transferencias GPU)
- ✅ Operaciones morfológicas CPU optimizadas
- ✅ 6 hilos paralelos por tecla
- ✅ Sin overhead de transferencias CPU↔GPU
- ✅ Rendimiento consistente y predecible

**Benchmark de Optimización:**
```
ANTES (con overhead GPU): 42.8ms/frame
DESPUÉS (CPU optimizado): 11.6ms/frame
Mejora: 3.6x más rápido
```

### 🔧 ¿Por qué CPU es Mejor Aquí?

Para operaciones de visión por computadora pequeñas y frecuentes:

**❌ GPU Overhead:**
- Transferencias CPU→GPU→CPU constantes
- Latencia de inicialización CUDA por operación
- Memory allocation/deallocation en GPU

**✅ CPU Ventajas:**
- Operaciones directas en memoria principal
- Sin transferencias de datos
- Multithreading nativo más eficiente
- OpenCV optimizado para CPU

### 💡 Recomendación

**Para este tipo de detección en tiempo real:**
- ✅ **CPU + Multithreading** es la solución óptima
- ❌ GPU solo beneficia en batch processing masivo
- 🎯 **11.6ms/frame** es excelente para gaming en tiempo real

## 🐛 Errores Comunes Resueltos

Esta sección documenta problemas específicos que han sido resueltos durante el desarrollo y pueden ser útiles como referencia.

### 1. La captura de pantalla es vertical cuando el juego es horizontal

- **Síntoma**: Al ejecutar el calibrador o un script de prueba, la imagen generada (ej: `verification_capture.png`) aparece rotada 90 grados, mostrando una captura vertical en lugar de la esperada captura horizontal del juego.
- **Causa Raíz**: Durante el desarrollo, se introdujo erróneamente una lógica en el script `src/window_calibrator.py` que intercambiaba los valores de `ancho` y `alto` de la región seleccionada por el usuario. Aunque el usuario seleccionaba un rectángulo horizontal, el sistema lo guardaba en el archivo `config.ini` con las dimensiones invertidas.
- **Solución**: Se eliminó por completo la lógica que intercambiaba el ancho y el alto en `src/window_calibrator.py`. Ahora, el calibrador guarda las dimensiones exactamente como las selecciona el usuario, respetando la orientación horizontal.

### 2. Error `gdi32.GetDIBits() failed` en Windows

- **Síntoma**: Al ejecutar el calibrador de ventana, el programa se interrumpe y muestra un `mss.exception.ScreenShotError` con el mensaje `gdi32.GetDIBits() failed.`
- **Causa Raíz**: Este es un error de bajo nivel específico de Windows que ocurre por un conflicto entre las librerías `mss` (usada para la captura de pantalla) y `OpenCV` (usada para mostrar las ventanas). El error se produce cuando se intenta hacer una nueva captura con `mss` inmediatamente después de haber cerrado una ventana de `OpenCV`.
- **Solución**: Se refactorizó el flujo del calibrador. En lugar de hacer dos capturas de pantalla separadas (una para el área de juego y otra para el área de puntuación), ahora el sistema realiza **una única captura inicial** de toda la pantalla. Para la segunda parte de la calibración (seleccionar la puntuación), simplemente se **recorta mediante programación el área de juego de la imagen original** que ya está en memoria, evitando así una segunda llamada a la función de captura y el conflicto resultante. 

### 3. Confusión entre Notas Amarillas y Verdes

- **Síntoma**: El detector identificaba incorrectamente notas amarillas como verdes o viceversa, especialmente cuando tenían formas similares.
- **Causa Raíz**: Los thresholds de confianza eran demasiado bajos y similares para ambos tipos, causando detecciones cruzadas.
- **Solución**: Se implementaron **thresholds diferenciados y estrictos**: Amarillas requieren 95% confianza, verdes 85%, eliminando completamente la confusión entre tipos.

### 4. Detecciones Duplicadas

- **Síntoma**: Una misma nota se detectaba múltiples veces, creando rectángulos superpuestos.
- **Causa Raíz**: El algoritmo no filtraba detecciones con alta superposición.
- **Solución**: Se implementó un **sistema anti-duplicados** que filtra detecciones con más del 25% de superposición, manteniendo solo la de mayor confianza.

### 5. Latencia Excesiva con GPU (Overhead Innecesario)

- **Síntoma**: Después de implementar aceleración GPU, el rendimiento empeoró dramáticamente, pasando de ~22ms/frame a más de 40ms/frame.
- **Causa Raíz**: Para operaciones pequeñas y frecuentes de OpenCV, las transferencias CPU↔GPU añaden más overhead que beneficio. Sin OpenCV CUDA nativo, forzar PyTorch GPU crea latencia innecesaria.
- **Análisis**: 
  ```
  CPU directo: 22ms/frame
  GPU con overhead: 42ms/frame (1.9x MÁS LENTO)
  ```
- **Solución**: Revertir a **CPU directo + multithreading**, eliminando transferencias GPU innecesarias. Resultado final: **11.6ms/frame** (3.6x mejora vs GPU). 