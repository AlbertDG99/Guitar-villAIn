# 🎸 Guitar Hero AI - Personal Research Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a **personal research project** exploring two different approaches to solve the same problem: real-time note detection in Guitar Hero using Computer Vision and AI techniques. The goal is to learn and experiment with different technologies and methodologies.

## 🎯 **Two Independent Research Approaches**

### **1. Color Pattern Approach** (`color_pattern_approach/`)
- **Purpose**: Real-time note detection and automatic key pressing
- **Technology**: Computer Vision + HSV Color Filtering
- **Learning Focus**: Computer Vision, Real-time Processing, Multi-threading
- **Status**: ✅ **Fully Functional**

### **2. Reinforcement AI Approach** (`reinforcement_ai_approach/`)
- **Purpose**: AI agent that learns to play Guitar Hero autonomously
- **Technology**: Deep Q-Learning (DQN) + Computer Vision
- **Learning Focus**: Reinforcement Learning, Neural Networks, Gymnasium
- **Status**: 🔬 **Research & Development**

Both approaches are **completely independent** and represent different learning paths for the same problem.

## ✨ Research Goals and Philosophy

This project explores:

- **Computer Vision**: Real-time image processing and color detection
- **Performance Optimization**: Multi-threading, micro-image processing, FPS optimization
- **Machine Learning**: Deep Q-Networks, experience replay, neural network architectures
- **System Design**: Modular architecture, configuration management, error handling
- **Real-time Systems**: Screen capture, input simulation, concurrent processing

## 🛠️ Research Tools and Experiments

### 1. Detection Visualizer (`utils/polygon_visualizer.py`)

Real-time visualization of the detection process over the game window.

**Research Features:**
- **Real-Time Detection**: HSV-based note detection with configurable ranges
- **Polygon Visualization**: Custom detection areas for each lane
- **Performance Metrics**: FPS monitoring and detection accuracy
- **View Modes**: Multiple visualization modes for debugging
- **Concurrent Processing**: Multi-threaded lane analysis

### 2. HSV Calibrator (`utils/static_hsv_calibrator_plus.py`)

Interactive tool for experimenting with HSV color ranges and morphological operations.

**Research Features:**
- **Real-Time Adjustment**: Interactive sliders for HSV parameters
- **Instant Preview**: Live feedback on parameter changes
- **Morphological Operations**: Experimentation with image processing techniques
- **Configuration Management**: Automatic saving of optimized parameters

## 🚀 Installation and Setup

### Prerequisites
- Python 3.11 or higher
- Windows 10/11 (optimized for Windows input simulation)

### Quick Setup

#### Color Pattern Approach
```bash
# Clone and setup
git clone <REPOSITORY_URL>
cd guitar_hero_ia/color_pattern_approach
pip install -r requirements.txt

# Run the main experiment
python color_pattern_visualizer.py
```

#### Reinforcement AI Approach
```bash
# Setup AI research environment
cd guitar_hero_ia/reinforcement_ai_approach
pip install -r requirements.txt

# Start training experiment
python train.py
```

## 📸 Research Demonstrations

*Note: Screenshots and GIFs would be added here to show the research results*

### 🎯 Real-time Detection Experiments
- **Performance**: 47.5 FPS average with HSV filtering
- **Accuracy**: Real-time note detection with configurable sensitivity
- **Optimization**: 44.2% area reduction using custom polygons

### 🧠 AI Learning Experiments
- **Architecture**: Dueling DQN with prioritized experience replay
- **Training**: Autonomous learning through trial and error
- **Metrics**: Reward progression and learning curves

### 🛠️ Calibration and Debugging Tools
- **HSV Calibration**: Interactive color range optimization
- **Polygon Visualization**: Real-time detection area monitoring
- **Performance Analysis**: FPS and accuracy metrics

## 📂 Project Structure

```
guitar_hero_ia/
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
├── data/
│   └── templates/
│       └── image.png               # Calibration template
│
├── lanzar_color_pattern.bat        # Quick start script
├── todo_list.md                    # Research progress tracking
└── README.md                       # This overview
```

## 🎯 Research Methodology

### **Computer Vision Approach**
1. **HSV Color Filtering**: Real-time color detection instead of template matching
2. **Micro-image Processing**: Processing only relevant screen areas for performance
3. **Multi-threading**: Parallel lane analysis to maintain high FPS
4. **Polygon Optimization**: Custom detection areas reducing processing by 44.2%

### **AI Approach**
1. **Deep Q-Network**: Neural network learning optimal actions
2. **Experience Replay**: Storing and replaying past experiences
3. **Prioritized Sampling**: Focusing on important experiences
4. **Dueling Architecture**: Separating value and advantage estimation

## 📊 Research Results

### **Performance Metrics**
- **FPS**: 47.5 average (excellent for real-time processing)
- **Detection Accuracy**: Configurable sensitivity for different scenarios
- **Memory Usage**: Optimized for minimal resource consumption
- **CPU Utilization**: Efficient multi-threading implementation

### **Technical Achievements**
- **HSV vs Template Matching**: 10x faster performance
- **Area Optimization**: 44.2% reduction in processing area
- **Real-time Processing**: Sub-21ms frame processing
- **Modular Architecture**: Independent, reusable components

## 🔧 Experimentation Guide

### **Color Pattern Experiments**
```bash
# Run main detection experiment
cd color_pattern_approach
python color_pattern_visualizer.py

# Calibrate HSV ranges
python -m utils.static_hsv_calibrator_plus

# Quick performance test
python quick_benchmark.py
```

### **AI Experiments**
```bash
# Start training experiment
cd reinforcement_ai_approach
python train.py

# Calibrate detection regions
python combo_calibrator.py

# Visualize detection
python utils/polygon_visualizer.py
```

## 📝 Research Notes

### **Key Learnings**
- **HSV Color Filtering**: Superior performance over template matching
- **Multi-threading**: Essential for real-time computer vision
- **Configuration Management**: Critical for reproducible experiments
- **Error Handling**: Robust systems require comprehensive error management

### **Technical Challenges**
- **Real-time Performance**: Balancing accuracy with speed
- **Color Calibration**: Adapting to different game versions
- **Input Simulation**: Reliable key pressing without detection
- **AI Training**: Stable learning in complex environments

## 🧠 Future Research Directions

### **Short Term**
- [ ] Enhanced note detection algorithms
- [ ] Improved AI training stability
- [ ] Additional calibration tools
- [ ] Performance benchmarking suite

### **Medium Term**
- [ ] Advanced AI models (Transformer-based)
- [ ] Multi-game support research
- [ ] Web-based experiment interface
- [ ] Comprehensive testing framework

### **Long Term**
- [ ] Real-time multiplayer AI competitions
- [ ] Advanced AI with human-like patterns
- [ ] Integration with streaming platforms
- [ ] Cross-platform compatibility research

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- PyTorch team for deep learning framework
- Guitar Hero community for inspiration
- Various research papers and tutorials that guided this learning journey

---

**🎸 A personal exploration of Computer Vision and AI applied to rhythm games!**