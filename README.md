# 🎸 Guitar-villAIn

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Personal research project exploring two independent approaches to automate key presses in Guitar Hero-like games using Computer Vision and, experimentally, Reinforcement Learning. The goal is to learn and experiment with real-time vision and system design.

## 🎯 Independent approaches

### 1) Color/Polygon Approach (`color_pattern_approach/`)
- **Idea**: Detect notes by HSV color inside lane polygons and simulate key presses.
- **Tech**: OpenCV, NumPy, MSS/PyAutoGUI, pydirectinput.
- **Design**: Dedicated capture thread, per-lane processing, anti-spam input logic.

### 2) Reinforcement Learning Approach (`reinforcement_ai_approach/`)
- **Idea**: A DQN agent learns to play from vision-based observations.
- **Tech**: Gymnasium, PyTorch (DQN/Dueling/Double), OpenCV.
- **Design**: `GuitarHeroEnv` environment, `DQNAgent` agent, threaded capture, combo/score OCR.

Both approaches are **completely independent** and represent different learning paths for the same problem.

## ✨ Key technologies

This project explores:

- **Vision**: OpenCV (HSV, morphology, contours, polygon masking).
- **Input**: `pydirectinput` to simulate keys on Windows.
- **Capture**: `mss` (preferred) and `pyautogui` (fallback) in a dedicated thread.
- **RL**: PyTorch, Gymnasium, DQN/Dueling/Double, Prioritized Replay (experimental).
- **OCR**: Tesseract via `pytesseract` for combo/score (RL approach).

## 🛠️ Project structure

```
Guitar-villAIn/
├── color_pattern_approach/
│   ├── color_pattern_visualizer.py
│   ├── screen_capture.py
│   ├── config_manager.py
│   ├── metrics.py
│   ├── config.ini
│   └── requirements.txt
├── reinforcement_ai_approach/
│   ├── src/
│   │   ├── ai/
│   │   │   ├── dqn_agent.py
│   │   │   └── env.py
│   │   ├── core/
│   │   │   ├── screen_capture.py
│   │   │   ├── score_detector.py
│   │   │   └── combo_detector.py
│   │   └── utils/
│   │       ├── config_manager.py
│   │       ├── helpers.py
│   │       └── logger.py
│   ├── utils/
│   │   ├── polygon_visualizer.py
│   │   └── static_hsv_calibrator_plus.py
│   ├── config/
│   │   └── config.ini
│   ├── train.py
│   └── requirements.txt
├── sloth_approach/
│   └── polygon_visualizer.py
└── lanzar_color_pattern.bat
```

## 🚀 Install & run

### Requirements
- Python 3.11+
- Windows 10/11 (Administrator privileges required for key presses)

### Run

Color/Polygon Approach
```powershell
# PowerShell as Administrator, from repo root
cd color_pattern_approach
pip install -r requirements.txt
python -m color_pattern_approach.color_pattern_visualizer
```

Reinforcement Learning Approach (experimental)
```powershell
# PowerShell as Administrator, from repo root
cd reinforcement_ai_approach
pip install -r requirements.txt
python train.py
```

## 🧭 Process diagrams

### Color/Polygon (a.k.a. Sloth) approach – per frame
```mermaid
flowchart TD
    A[MSS thread capture] -->|latest frame| B[Main loop]
    B --> C[Prepare per-lane tasks]
    C --> D{ThreadPoolExecutor}
    D --> E1[Lane S: crop microimage -> HSV -> morphology -> contours]
    D --> E2[Lane D]
    D --> E3[Lane F]
    D --> E4[Lane J]
    D --> E5[Lane K]
    D --> E6[Lane L]
    E1 --> F[Filter by polygon and area]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    E6 --> F
    F --> G[Aggregate detections]
    G --> H[Input logic: green/yellow + anti-spam]
    H --> I[Visual overlay & HUD]
    I --> J[Next frame]
```

Notes:
- Each lane is processed independently in parallel.
- For green notes, yellow processing can be skipped within the same lane.
- Input logic applies cooldowns and controlled randomness.

### Reinforcement Learning approach – per step
```mermaid
flowchart TD
    A[env.step(action)] --> B[Apply action (key or no-op)]
    B --> C[Get latest frame]
    C --> D[HSV masks + polygons -> 6-lane state]
    D --> E[OCR combo/score in parallel]
    E --> F[Compute reward]
    F --> G[Return obs/reward/done]
    G --> H[DQN agent: select_action/train]
```

## 📂 Project Structure

```
Guitar-villAIn/
├── color_pattern_approach/          # Computer Vision Research
│   ├── color_pattern_visualizer.py  # Main detection experiment
│   ├── config_manager.py           # Configuration system
│   ├── screen_capture.py           # Optimized capture system
│   ├── config.ini                  # Experiment parameters
│   ├── requirements.txt            # Dependencies
│   └── README.md                   # Research documentation
│
├── reinforcement_ai_approach/       # AI Research
│   ├── src/
│   │   ├── ai/
│   │   │   ├── dqn_agent.py        # Deep Q-Network implementation
│   │   │   └── env.py              # Gymnasium environment
│   │   ├── core/
│   │   │   ├── combo_detector.py   # OCR-based combo detection
│   │   │   ├── score_detector.py   # Score detection system
│   │   │   └── screen_capture.py   # Screen capture for AI
│   │   └── utils/
│   │       ├── config_manager.py   # Configuration management
│   │       ├── helpers.py          # Utility functions
│   │       └── logger.py           # Logging system
│   ├── utils/
│   │   ├── polygon_visualizer.py   # Detection visualization
│   │   └── static_hsv_calibrator_plus.py # HSV calibration
│   ├── config/
│   │   └── config.ini              # AI experiment parameters
│   ├── train.py                    # Training experiment
│   ├── combo_calibrator.py         # Calibration tool
│   ├── requirements.txt            # AI dependencies
│   └── README.md                   # AI research documentation
│
├── lanzar_color_pattern.bat        # Quick start script
└── README.md                       # This overview
```

## 🔧 Configuration
- Each approach has its own `config.ini` defining capture, lane polygons, HSV ranges and auxiliary parameters.
- The RL approach additionally defines agent hyperparameters and OCR regions.

## 🛠️ Tools
- Detection visualizer (both approaches) to debug polygons and masks.
- HSV calibrator (RL approach) to tune ranges and morphology.
 - Setup wizard (RL approach) to quickly reconfigure capture/polygons/ROIs.

## 🧪 Quick guide

Color/Polygon
```powershell
cd color_pattern_approach
python -m color_pattern_approach.color_pattern_visualizer
```

RL (experimental)
```powershell
cd reinforcement_ai_approach
python train.py
python utils/polygon_visualizer.py
```

## 📷 Screenshots
Place images in `assets/screenshots/`. Example usage in docs:

```markdown
![P5X rhythm minigame](assets/screenshots/p5x_rhythm_example.png)
```

## 🧠 Future work
- Color approach: better morphology heuristics and per-lane segmentation.
- RL approach: stabilize training and improve observation/reward design.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community
- PyTorch team
- Guitar Hero community

---

**🎸 Personal project of Computer Vision and AI applied to rhythm games.**