# 🧠 Reinforcement AI Approach

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)

Advanced reinforcement learning system for Guitar Hero automation using Deep Q-Networks (DQN) and modern AI techniques.

## 🎯 Overview

This subproject implements a sophisticated reinforcement learning agent that learns to play Guitar Hero autonomously. It uses state-of-the-art techniques including:

- **Deep Q-Networks (DQN)** with Double and Dueling variants
- **Prioritized Experience Replay** for efficient learning
- **Mixed Precision Training** for GPU optimization
- **Real-time Computer Vision** for note detection
- **Gymnasium Environment** for standardized RL interface

## 🏗️ Architecture

### Core Components

```
reinforcement_ai_approach/
├── src/
│   ├── ai/                    # AI and RL components
│   │   ├── dqn_agent.py      # Deep Q-Network agent
│   │   └── env.py            # Gymnasium environment
│   ├── core/                  # Core detection systems
│   │   ├── combo_detector.py  # Combo multiplier detection
│   │   ├── score_detector.py  # Score reading via OCR
│   │   └── screen_capture.py  # Optimized frame capture
│   └── utils/                 # Utility modules
│       ├── config_manager.py  # Configuration management
│       ├── helpers.py         # Helper functions
│       └── logger.py          # Logging system
├── utils/                     # Development tools
│   ├── polygon_visualizer.py  # Real-time detection visualizer
│   └── static_hsv_calibrator_plus.py  # HSV calibration tool
├── config/                    # Configuration files
│   └── config.ini            # AI and system parameters
├── models/                    # Trained model storage
├── train.py                   # Training orchestration
├── combo_calibrator.py        # Combo region calibration
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the reinforcement AI approach
cd reinforcement_ai_approach

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with CUDA support for GPU acceleration
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Basic Usage

```bash
# Start training the AI agent
python train.py

# Calibrate combo detection region
python combo_calibrator.py

# Visualize detection in real-time
python utils/polygon_visualizer.py
```

## 🧠 How It Works

### 1. Environment Interface (`src/ai/env.py`)

The `GuitarHeroEnv` class provides a standardized interface between the AI and the game:

```python
# Example usage
env = GuitarHeroEnv()
state = env.reset()

for episode in range(1000):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    agent.store_experience(state, action, reward, next_state, done)
    state = next_state
    
    if done:
        state = env.reset()
```

**Key Features:**
- **State Space**: 6-lane note detection vector
- **Action Space**: 7 discrete actions (no-op + 6 keys)
- **Reward System**: Score-based rewards with combo multipliers
- **Real-time Processing**: 60+ FPS detection and action execution

### 2. Deep Q-Network Agent (`src/ai/dqn_agent.py`)

Advanced DQN implementation with multiple optimizations:

```python
# Initialize agent
agent = DQNAgent(
    state_size=6,
    action_size=7,
    learning_rate=1e-4,
    epsilon=1.0,
    epsilon_decay=0.995
)

# Training loop
for step in range(max_steps):
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.store_experience(state, action, reward, next_state, done)
    agent.train_step()
    
    if step % target_update_freq == 0:
        agent.update_target_network()
```

**Advanced Features:**
- **Double DQN**: Reduces Q-value overestimation
- **Dueling DQN**: Separates value and advantage estimation
- **Prioritized Replay**: Efficient experience sampling
- **Mixed Precision**: GPU acceleration with automatic scaling

### 3. Computer Vision Pipeline

Real-time note detection using optimized computer vision:

```python
# HSV-based color detection
def detect_notes(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for yellow and green notes
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    return process_contours(yellow_mask, green_mask)
```

**Performance Optimizations:**
- **Micro-image Processing**: Only analyze relevant screen regions
- **Multithreading**: Parallel lane analysis
- **MSS Capture**: High-performance screen capture
- **GPU Acceleration**: CUDA-enabled PyTorch operations

## ⚙️ Configuration

### AI Parameters (`config/config.ini`)

```ini
[AI]
# Learning Parameters
learning_rate = 0.0001
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 100000
target_update_frequency = 1000

# Architecture Options
double_dqn = true
dueling_dqn = true
prioritized_replay = true
use_mixed_precision = true

# Training Configuration
model_save_path = models/guitar_hero_agent.pth
num_episodes = 10000
max_steps_per_episode = 5000
save_frequency = 100
```

### HSV Color Ranges

```ini
[HSV_RANGES]
# Yellow notes
yellow_h_min = 15
yellow_s_min = 100
yellow_v_min = 100
yellow_h_max = 40
yellow_s_max = 255
yellow_v_max = 255

# Green notes
green_h_min = 25
green_s_min = 40
green_v_min = 40
green_h_max = 95
green_s_max = 255
green_v_max = 255
```

## 🎮 Training Process

### Phase 1: Environment Setup
1. **Calibrate Detection Regions**: Use `combo_calibrator.py` to set up score detection
2. **Adjust HSV Ranges**: Use `utils/static_hsv_calibrator_plus.py` for optimal color detection
3. **Verify Detection**: Use `utils/polygon_visualizer.py` to confirm note detection accuracy

### Phase 2: Agent Training
1. **Initialize Agent**: Create DQN agent with configured parameters
2. **Training Loop**: Run episodes with exploration and learning
3. **Model Checkpoints**: Save progress periodically
4. **Performance Monitoring**: Track rewards, accuracy, and learning curves

### Phase 3: Evaluation
1. **Test Performance**: Evaluate trained agent on unseen songs
2. **Fine-tuning**: Adjust hyperparameters based on performance
3. **Deployment**: Use trained model for autonomous gameplay

## 📊 Performance Metrics

### Training Metrics
- **Episode Reward**: Average score per episode
- **Learning Rate**: Improvement over time
- **Epsilon Decay**: Exploration vs exploitation balance
- **Loss Function**: Q-value prediction accuracy

### Runtime Metrics
- **Detection FPS**: Notes detected per second
- **Action Latency**: Time from detection to key press
- **Accuracy**: Percentage of notes correctly played
- **Combo Multiplier**: Average combo maintained

## 🛠️ Development Tools

### Visualization Tools

#### Real-time Detection Visualizer
```bash
python utils/polygon_visualizer.py
```
- Shows live note detection
- Displays FPS and performance metrics
- Multiple view modes (normal, masks, debug)

#### HSV Calibrator
```bash
python utils/static_hsv_calibrator_plus.py
```
- Interactive HSV range adjustment
- Real-time preview of color detection
- Save optimized parameters

### Calibration Tools

#### Combo Region Calibrator
```bash
python combo_calibrator.py
```
- Mouse-based region selection
- Automatic configuration saving
- Visual feedback for accuracy

## 🔧 Advanced Features

### Mixed Precision Training
```python
# Automatic mixed precision for GPU acceleration
from torch.cuda.amp import GradScaler

scaler = GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(predictions, targets)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Prioritized Experience Replay
```python
# Efficient experience sampling based on TD-error
class PrioritizedReplayBuffer:
    def sample(self, batch_size):
        # Sample experiences with higher TD-error
        priorities = self.get_priorities()
        indices = np.random.choice(len(self), batch_size, p=priorities)
        return self.get_batch(indices)
```

### Dueling DQN Architecture
```python
class DuelingDQN(nn.Module):
    def forward(self, x):
        features = self.feature_layer(x)
        
        # Value stream
        value = self.value_stream(features)
        
        # Advantage stream
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the correct directory
cd reinforcement_ai_approach

# Install missing dependencies
pip install -r requirements.txt
```

#### Detection Issues
```bash
# Recalibrate HSV ranges
python utils/static_hsv_calibrator_plus.py

# Check detection regions
python utils/polygon_visualizer.py
```

#### Training Problems
```python
# Reduce learning rate if training is unstable
learning_rate = 0.00001

# Increase epsilon decay for more exploration
epsilon_decay = 0.999

# Check GPU memory usage
torch.cuda.empty_cache()
```

## 📈 Future Enhancements

### Planned Features
- **Transformer-based Models**: Attention mechanisms for better sequence learning
- **Multi-agent Training**: Competitive and cooperative learning
- **Adaptive Difficulty**: Dynamic difficulty adjustment
- **Real-time Adaptation**: Online learning during gameplay

### Research Directions
- **Meta-learning**: Learning to learn new songs quickly
- **Hierarchical RL**: High-level strategy and low-level execution
- **Imitation Learning**: Learning from human demonstrations
- **Multi-modal Input**: Audio + visual fusion

## 🤝 Contributing

See the main project README for contribution guidelines. For AI-specific contributions:

1. **Test on Multiple Songs**: Ensure generalization across different difficulties
2. **Benchmark Performance**: Compare against baseline implementations
3. **Document Changes**: Update configuration and documentation
4. **Validate Results**: Ensure reproducible training outcomes

## 📚 References

- [Deep Q-Networks](https://arxiv.org/abs/1312.5602)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**🧠 Advanced AI system for autonomous Guitar Hero gameplay using state-of-the-art reinforcement learning techniques!** 