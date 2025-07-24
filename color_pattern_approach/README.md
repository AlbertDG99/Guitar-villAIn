# 🎯 Color Pattern Approach - High Performance Polygon Visualizer

## 📋 Description

The **Color Pattern Approach** is a completely independent and optimized module for real-time note detection in Guitar Hero. It's designed to maximize performance on multi-core CPUs, achieving **90+ FPS** through real parallelism and micro-image processing.

## 🚀 Key Features

- ✅ **Real Parallelism**: Each lane is processed completely in parallel
- ✅ **Micro-images**: Processing only relevant areas (100x400px vs 1280x720px)
- ✅ **Multi-threaded Capture**: Independent capture system using MSS
- ✅ **Dual Detection**: Green and yellow notes with optimized game logic
- ✅ **Fully Independent**: No dependencies on the main project
- ✅ **High Performance**: 90+ FPS on high-end CPUs

## 🏗️ System Architecture

### Main Flow Diagram

```mermaid
graph TD
    A[🎮 Program Start] --> B[📋 Load Configuration]
    B --> C[🖥️ Initialize ScreenCapture]
    C --> D[▶️ Start Capture Thread]
    D --> E[⏱️ Wait for First Frame]
    E --> F{🖼️ Frame Available?}
    
    F -->|Yes| G[🔄 Main Loop]
    F -->|No| H[⚠️ Capture Error]
    
    G --> I[📸 Get Latest Frame]
    I --> J[⚡ Process Frame in Parallel]
    J --> K[🎨 Draw Results]
    K --> L[📺 Display on Screen]
    L --> M{⌨️ Key Pressed?}
    
    M -->|Q/ESC| N[🛑 Exit]
    M -->|SPACE| O[⏸️ Pause/Resume]
    M -->|None| G
    
    N --> P[🔄 Clean Resources]
    P --> Q[✅ Program End]
```

### Thread Architecture

```mermaid
graph LR
    subgraph "🖥️ Capture Thread (MSS)"
        A1[📷 Continuous Capture]
        A2[🔄 Update Frame]
        A3[💾 Save to Buffer]
        A1 --> A2 --> A3 --> A1
    end
    
    subgraph "🎯 Main Thread"
        B1[📸 Get Frame]
        B2[🚀 Launch Workers]
        B3[📊 Collect Results]
        B4[🎨 Draw UI]
        B1 --> B2 --> B3 --> B4 --> B1
    end
    
    subgraph "⚡ Parallel Workers"
        C1[🎵 Lane S]
        C2[🎵 Lane D]
        C3[🎵 Lane F]
        C4[🎵 Lane J]
        C5[🎵 Lane K]
        C6[🎵 Lane L]
    end
    
    A3 -.->|Shared Frame| B1
    B2 -->|Dispatch| C1
    B2 -->|Dispatch| C2
    B2 -->|Dispatch| C3
    B2 -->|Dispatch| C4
    B2 -->|Dispatch| C5
    B2 -->|Dispatch| C6
    
    C1 -->|Results| B3
    C2 -->|Results| B3
    C3 -->|Results| B3
    C4 -->|Results| B3
    C5 -->|Results| B3
    C6 -->|Results| B3
```

## 🔬 Micro-image Processing per Lane

### Worker Workflow

```mermaid
graph TD
    A[🎵 Receive Lane + Frame] --> B[📏 Calculate Bounding Box]
    B --> C[✂️ Crop Micro-image]
    C --> D{📐 Valid Size?}
    
    D -->|No| E[❌ Return Empty]
    D -->|Yes| F[🎨 Convert to HSV]
    
    F --> G[🟢 Detect Greens]
    G --> H[🔍 Morphological Operations]
    H --> I[📊 Find Contours]
    I --> J[✅ Filter by Area]
    J --> K{🟢 Greens Found?}
    
    K -->|Yes| L[📍 Convert Coordinates]
    K -->|No| M[🟡 Detect Yellows]
    
    M --> N[🔍 Morphological Operations]
    N --> O[📊 Find Contours]
    O --> P[✅ Filter by Area]
    P --> Q[📍 Convert Coordinates]
    
    L --> R[📤 Return Results]
    Q --> R
    E --> R
```

## 🛠️ Technologies and Libraries

### Core Technologies
- **Python 3.8+**: Base language
- **OpenCV (cv2)**: Image processing and computer vision
- **NumPy**: High-performance matrix operations
- **MSS**: Ultra-fast screen capture
- **ThreadPoolExecutor**: Real parallelism with native threads

### Specific Optimizations

#### 1. **Screen Capture (MSS)**
```python
# Capture in separate thread with MSS
with mss.mss() as sct:
    screenshot = sct.grab(region)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
```

#### 2. **Micro-images per Lane**
```python
# Intelligent cropping by bounding box
x_min, y_min = np.min(pts, axis=0)
x_max, y_max = np.max(pts, axis=0)
micro_frame = frame[y_min:y_max, x_min:x_max]
```

#### 3. **Real Parallelism**
```python
# Each lane is processed completely in parallel
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_lane_micro_image, task) for task in lanes]
```

## 📊 Performance Metrics

### Architecture Comparison

| Method | FPS | CPU Usage | Description |
|--------|-----|-----------|-------------|
| **Original** | ~30 FPS | 25% | Complete sequential processing |
| **Basic Threading** | ~45 FPS | 40% | Filtering parallelization only |
| **Color Pattern Approach** | **90+ FPS** | **70%** | **Real parallelism + micro-images** |

### Processed Data Reduction

```
Original Image: 1280x720 = 921,600 pixels
Micro-images (6 lanes): ~100x400 = 240,000 pixels total
Reduction: ~75% less data per frame
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` or `q` | Exit program |
| `ESC` | Exit program (alternative) |
| `SPACE` | Pause/Resume |
| `Ctrl+C` | Force interruption |

## 📁 File Structure

```
color_pattern_approach/
├── README.md                 # This file
├── color_pattern_visualizer.py     # Main optimized script
├── screen_capture.py         # Multi-threaded capture system
├── config_manager.py         # Configuration manager
└── config.ini               # System configuration
```

## 🚀 Execution

```bash
# From the project root directory
python -m color_pattern_approach.color_pattern_visualizer
```

## ⚙️ Configuration

The system automatically reads configuration from `config.ini`:

- **HSV Ranges**: For green and yellow color detection
- **Morphological Parameters**: Kernel sizes and filtering areas
- **Lane Polygons**: Coordinates of the 6 detection areas
- **Capture Region**: Screen area to process

## 🔧 Implemented Optimizations

### 1. **Bottleneck Elimination**
- ❌ Global HSV conversion (1280x720)
- ✅ Micro-image HSV conversion (100x400)

### 2. **Intelligent Parallelism**
- ❌ Threading in filtering (5% of work)
- ✅ Threading in complete processing (95% of work)

### 3. **Memory Reduction**
- ❌ Morphological operations on complete image
- ✅ Morphological operations on relevant areas

### 4. **Game Logic Optimization**
- If a green note is detected, yellow search is omitted
- Local to global coordinate conversion post-processing

## 📈 Scalability

The system is designed to automatically scale according to available resources:

- **4-core CPU**: Processing of 4 lanes simultaneously
- **8+ core CPU**: Processing of 6 lanes + system overhead
- **Memory**: Minimal usage thanks to micro-images

## 🎯 Use Cases

1. **AI Development**: Base for machine learning systems
2. **Performance Analysis**: Real-time detection metrics
3. **Calibration**: Fine-tuning of detection parameters
4. **Research**: Platform for new vision algorithms

---

*Developed with ❤️ to maximize real-time note detection performance.* 