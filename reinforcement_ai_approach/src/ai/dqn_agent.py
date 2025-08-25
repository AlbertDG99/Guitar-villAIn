"""
Advanced Deep Q-Network (DQN) agent for Guitar Hero AI.
"""

import random
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from src.utils.config_manager import ConfigManager

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class DQN(nn.Module):
    """Standard Deep Q-Network architecture."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture separating value and advantage streams."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature_layer = nn.Linear(state_size, 128)
        self.value_stream = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, 128)
        self.advantage_head = nn.Linear(128, action_size)

    def forward(self, x):
        features = F.relu(self.feature_layer(x))
        
        # Value stream
        value = F.relu(self.value_stream(features))
        value = self.value_head(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_stream(features))
        advantage = self.advantage_head(advantage)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:  # pylint: disable=too-many-instance-attributes
    """Prioritized Experience Replay buffer for efficient learning."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(
            state,
            action,
            reward,
            next_state,
            done,
            self.max_priority)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritization"""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])

        # Calculate probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities)

        # Calculate importance weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:  # pylint: disable=too-many-instance-attributes
    """Advanced DQN agent for Guitar Hero"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.config_manager = ConfigManager(config_path=str(Path(__file__).parents[2] / 'config' / 'config.ini'))
        config = self.config_manager.get_ai_config()

        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.99)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_frequency', 10)
        self.gamma = 0.99

        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.is_training = True

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)

        self.dueling_dqn = config.get('dueling_dqn', True)
        self.double_dqn = config.get('double_dqn', True)
        self.prioritized_replay = config.get('prioritized_replay', True)

        NetworkModule = DuelingDQN if self.dueling_dqn else DQN
        self.policy_net = NetworkModule(
            state_size, action_size).to(
            self.device)
        self.target_net = NetworkModule(
            state_size, action_size).to(
            self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.learning_rate)
        # Use per-sample loss for importance-sampling weights
        self.criterion = nn.SmoothL1Loss(reduction='none')

        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                capacity=config.get('memory_size', 100000))
        else:
            self.memory = []

        self.step_count = 0
        self.grad_scaler = GradScaler() if self.use_mixed_precision else None

    def set_train_mode(self):
        """Set agent to training mode."""
        self.is_training = True
        self.policy_net.train()

    def set_eval_mode(self):
        """Set agent to evaluation mode."""
        self.is_training = False
        self.policy_net.eval()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if self.is_training and random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if self.prioritized_replay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None

        if self.prioritized_replay:
            batch, indices, weights = self.memory.sample(self.batch_size)
            if not batch:
                return None
            
            states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
            weights = torch.ones(self.batch_size).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selection on policy net, evaluation on target net
                next_actions = self.policy_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN: max over target net
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]

        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones.unsqueeze(1)))

        # Per-sample MSE for weighting
        loss_per_sample = self.criterion(current_q_values, target_q_values)
        # If gather produced (batch,1), reduce last dim
        loss_per_sample = loss_per_sample.view(-1)
        loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad()
        
        if self.use_mixed_precision and self.grad_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.prioritized_replay:
            with torch.no_grad():
                td_errors = (current_q_values - target_q_values).abs().view(-1).cpu().numpy()
                self.memory.update_priorities(indices, td_errors)

        self.step_count += 1

        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)

    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'memory_size': len(self.memory),
            'device': str(self.device)
        }
