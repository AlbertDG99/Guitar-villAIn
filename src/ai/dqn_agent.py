"""
DQN Agent - Agente Deep Q-Learning Avanzado
===========================================

Agente de Deep Q-Learning optimizado para RTX 5090 con características avanzadas.
"""

import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union, Deque
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.utils.logger import setup_logger, performance_logger
from src.utils.config_manager import ConfigManager


# Estructura para almacenar experiencias
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'priority']
)


# Definición de la red neuronal estándar (DQN)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    """Red neuronal Dueling DQN con Multi-Head para acciones multidimensionales."""

    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class PrioritizedReplayBuffer:  # pylint: disable=too-many-instance-attributes
    """Buffer de experiencia con priorización"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        """Devuelve el tamaño actual del buffer."""
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        """Añadir experiencia al buffer"""
        experience = Experience(state, action, reward, next_state, done, self.max_priority)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Muestrear batch con priorización"""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])

        # Calcular probabilidades
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Muestrear índices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calcular pesos de importancia
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Extraer experiencias
        batch = [self.buffer[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Actualizar prioridades basadas en TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:  # pylint: disable=too-many-instance-attributes
    """Agente DQN avanzado para Guitar Hero"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.config_manager = ConfigManager(config_path='config/config.ini')
        config = self.config_manager.get_ai_config()

        # Parámetros extraídos desde el diccionario de configuración
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.99)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_frequency', 10)
        self.gamma = 0.99  # Discount factor

        # Configuraciones avanzadas
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.is_training = True

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)
        
        self.dueling_dqn = config.get('dueling_dqn', True)
        self.prioritized_replay = config.get('prioritized_replay', True)

        # Redes neuronales
        NetworkModule = DuelingDQN if self.dueling_dqn else DQN
        self.policy_net = NetworkModule(state_size, action_size).to(self.device)
        self.target_net = NetworkModule(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.logger = setup_logger('DQNAgent')
        self.logger.info("DQN Agent initialized on %s", self.device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.5)

        # Buffer de experiencia
        memory_size = config.get('memory_size', 10000)
        self.memory: Union[PrioritizedReplayBuffer, Deque]
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

        # GradScaler para precisión mixta
        self.scaler = GradScaler(enabled=self.use_mixed_precision)
        
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

    def set_train_mode(self):
        """Pone el agente en modo entrenamiento."""
        self.is_training = True
        self.policy_net.train()
        self.logger.info("Agente en modo entrenamiento.")

    def set_eval_mode(self):
        """Pone el agente en modo evaluación."""
        self.is_training = False
        self.policy_net.eval()
        self.logger.info("Agente en modo evaluación.")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Selecciona una acción (vector de 0s y 1s) usando epsilon-greedy."""
        if self.is_training and random.random() < self.epsilon:
            # Acción aleatoria: un vector binario del tamaño correcto
            action = np.random.randint(0, 2, size=self.action_size, dtype=np.int64)
            return np.atleast_1d(action)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # Acción basada en política: pulsar si el Q-value es positivo
            action = (q_values > 0).long().cpu().numpy().flatten()
            return action

    def store_experience(self, state, action, reward, next_state, done):
        """Guardar experiencia en el buffer de memoria"""
        if self.memory is None:
            return
        
        if self.prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.add(state, action, reward, next_state, done)
        elif not self.prioritized_replay and isinstance(self.memory, deque):
            experience = Experience(state, action, reward, next_state, done, 0)
            self.memory.append(experience)

    def train_step(self) -> Optional[float]:
        """Realizar un paso de entrenamiento"""
        if self.memory is None:
            return None
            if len(self.memory) < self.batch_size:
                return None

        experiences: List[Experience]
        indices: Optional[np.ndarray] = None
        weights: torch.Tensor

        if self.prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            experiences, indices, np_weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(np_weights).unsqueeze(1).to(self.device)
        elif not self.prioritized_replay and isinstance(self.memory, deque):
            experiences = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size, 1).to(self.device)
        else:
            # Caso inesperado
                return None

        # Extraer datos del batch
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            # 1. Obtener Q-values para el estado actual y la acción tomada
            q_values = self.policy_net(states)
            # Para MultiBinary, el Q-value de la acción es la suma de los Q-values de las teclas pulsadas
            q_values_selected = (q_values * actions).sum(dim=1)

            # 2. Obtener Q-values del siguiente estado para el target
            with torch.no_grad():
                next_q_values_target = self.target_net(next_states)
                
                if self.dueling_dqn:
                    # Double DQN: Usar la red principal para elegir las mejores acciones futuras
                    next_q_values_policy = self.policy_net(next_states)
                    # La mejor acción es pulsar si el Q-value de la policy net es positivo
                    next_actions = (next_q_values_policy > 0).long()
                    # El Q-value se obtiene de la target net usando esas acciones
                    next_q_values_selected = (next_q_values_target * next_actions).sum(dim=1)
                else:
                    # DQN estándar: La mejor acción y el valor vienen de la target net
                    # El valor es la suma de todos los Q-values positivos
                    next_q_values_selected = torch.clamp(next_q_values_target, min=0).sum(dim=1)

            # Calcular el valor Q objetivo (y = r + gamma * Q_target(s', a_best))
            expected_q_values = rewards + (self.gamma * next_q_values_selected * (1 - dones))
            
        # Calcular la pérdida
        loss = F.mse_loss(q_values_selected, expected_q_values)

        # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        self.training_step += 1
        self.losses.append(loss.item())

        if self.prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            td_errors = (expected_q_values - q_values_selected).abs().detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors)
        
        return loss.item()

    def update_target_network(self):
        """Actualizar la red objetivo"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Red objetivo actualizada.")

    def decay_epsilon(self):
        """Reducir epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        """Guardar modelo entrenado"""
        self.logger.info("Guardando modelo en %s...", path)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, path)
        self.logger.info("Modelo guardado.")

    def load_model(self, path: str):
        """Cargar modelo entrenado"""
        self.logger.info("Cargando modelo desde %s...", path)
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'state_size' in checkpoint and checkpoint['state_size'] != self.state_size:
            self.logger.warning("El tamaño del estado del modelo guardado no coincide. Puede haber errores.")
        if 'action_size' in checkpoint and checkpoint['action_size'] != self.action_size:
            self.logger.warning("El tamaño de la acción del modelo guardado no coincide. Puede haber errores.")

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_step = checkpoint.get('training_step', 0)
        self.logger.info("Modelo cargado exitosamente.")

    def get_stats(self) -> Dict:
        """Obtener estadísticas de entrenamiento"""
        return {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory) if self.prioritized_replay else len(self.memory),
            'average_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }
