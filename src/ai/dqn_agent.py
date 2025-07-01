"""
DQN Agent - Agente Deep Q-Learning Avanzado
===========================================

Agente de Deep Q-Learning optimizado para RTX 5090 con características avanzadas.
"""

import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from src.utils.logger import setup_logger, performance_logger


# Estructura para almacenar experiencias
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'priority']
)


class DuelingDQN(nn.Module):
    """Red neuronal Dueling DQN optimizada para Guitar Hero"""

    def __init__(self, input_channels: int = 3, num_actions: int = 6, hidden_size: int = 512):
        super().__init__()

        # CNN para procesar imagen de pantalla
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Cuarta capa convolucional para más detalle
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        # Calcular tamaño después de convoluciones (para input 84x84x3)
        self.conv_output_size = self._get_conv_output_size(input_channels, 84, 84)

        # Capas totalmente conectadas para características adicionales
        self.feature_fc = nn.Sequential(
            nn.Linear(6 + 4, 128),  # 6 lanes + timing info
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Combinar features visuales y de juego
        combined_size = self.conv_output_size + 256

        # Dueling DQN: Separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # State value
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)  # Action advantages
        )

    def _get_conv_output_size(self, channels: int, height: int, width: int) -> int:
        """Calcular tamaño de salida de las capas convolucionales"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()

    def forward(self, screen_state: torch.Tensor, game_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red

        Args:
            screen_state: Tensor de imagen (batch_size, channels, height, width)
            game_features: Features del juego (batch_size, feature_size)
        """
        # Procesar imagen
        conv_out = self.conv_layers(screen_state)
        conv_flat = conv_out.view(conv_out.size(0), -1)

        # Procesar features del juego
        feature_out = self.feature_fc(game_features)

        # Combinar features
        combined = torch.cat([conv_flat, feature_out], dim=1)

        # Dueling streams
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)

        # Combinar value y advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


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

    def __init__(self, config, state_shape: Tuple[int, int, int] = (3, 84, 84),
                 num_actions: int = 6, feature_size: int = 10):
        self.config = config
        self.logger = setup_logger('DQNAgent')

        # Configuración del dispositivo (RTX 5090)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.logger.info("Using GPU: %s", torch.cuda.get_device_name())
            props = torch.cuda.get_device_properties(0)
            self.logger.info("CUDA Memory: %.1f GB", props.total_memory / 1e9)

        # Parámetros de entrenamiento
        ai_config = config.get_ai_config()
        self.learning_rate = ai_config['learning_rate']
        self.epsilon = ai_config['epsilon']
        self.epsilon_decay = ai_config['epsilon_decay']
        self.epsilon_min = ai_config['epsilon_min']
        self.batch_size = ai_config['batch_size']
        self.target_update_freq = ai_config['target_update_frequency']
        self.gamma = 0.99  # Discount factor

        # Configuraciones avanzadas
        self.double_dqn = config.getboolean('AI', 'double_dqn', True)
        self.dueling_dqn = config.getboolean('AI', 'dueling_dqn', True)
        self.prioritized_replay = config.getboolean('AI', 'prioritized_replay', True)
        self.use_mixed_precision = config.getboolean('AI', 'use_mixed_precision', True)

        # Redes neuronales
        self.q_network = DuelingDQN(state_shape[0], num_actions).to(self.device)
        self.target_network = DuelingDQN(state_shape[0], num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # El target network no se entrena directamente

        # Optimizador con configuración avanzada
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # Scheduler de learning rate
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.9
        )

        # Buffer de experiencia
        self.memory: Union[PrioritizedReplayBuffer, deque]
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(ai_config['memory_size'])
        else:
            self.memory = deque(maxlen=ai_config['memory_size'])

        # Mixed precision training
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.use_mixed_precision and self.device.type == 'cuda':
            # Suprimir advertencia de deprecación para GradScaler
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self.scaler = torch.cuda.amp.GradScaler()

        # Estadísticas
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

        self.is_training = True # Flag para modo entrenamiento/evaluación

        self.logger.info("DQN Agent initialized on %s", self.device)
        num_params = sum(p.numel() for p in self.q_network.parameters())
        self.logger.info("Network parameters: %s", f"{num_params:,}")

    def set_train_mode(self):
        """Pone el agente en modo entrenamiento."""
        self.is_training = True
        self.q_network.train()
        self.logger.info("Agente en modo entrenamiento.")

    def set_eval_mode(self):
        """Pone el agente en modo evaluación."""
        self.is_training = False
        self.q_network.eval()
        self.logger.info("Agente en modo evaluación.")

    def get_action(self, state: Dict, training: bool = True) -> int:
        """
        Seleccionar acción usando epsilon-greedy
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, 5)  # 6 acciones: 5 notas + no hacer nada

        with torch.no_grad():
            screen_tensor = torch.FloatTensor(state['screen']).unsqueeze(0).to(self.device)
            feature_tensor = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
            
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    q_values = self.q_network(screen_tensor, feature_tensor)
            else:
                q_values = self.q_network(screen_tensor, feature_tensor)
                
            return q_values.argmax().item()

    select_action = get_action  # Alias

    def store_experience(self, state, action, reward, next_state, done):
        """Almacenar experiencia en el buffer"""
        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append(Experience(state, action, reward, next_state, done, 1.0))

    def train_step(self) -> Optional[float]:
        """Realizar un paso de entrenamiento"""
        if isinstance(self.memory, PrioritizedReplayBuffer):
            if len(self.memory) < self.batch_size:
                return None
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return None
            batch = random.sample(list(self.memory), self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        # Preparar batch
        states_screen = torch.FloatTensor(np.array([e.state['screen'] for e in batch])).to(self.device)
        states_features = torch.FloatTensor(np.array([e.state['features'] for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states_screen = torch.FloatTensor(np.array([e.next_state['screen'] for e in batch])).to(self.device)
        next_states_features = torch.FloatTensor(np.array([e.next_state['features'] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)

        # Forward pass con mixed precision si está activado
        autocast_enabled = self.use_mixed_precision and self.device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            current_q_values = self.q_network(states_screen, states_features).gather(1, actions.unsqueeze(1))

            # Double DQN
            with torch.no_grad():
                if self.double_dqn:
                    next_actions = self.q_network(next_states_screen, next_states_features).argmax(1).unsqueeze(1)
                    next_q_values = self.target_network(next_states_screen, next_states_features).gather(1, next_actions)
                else:
                    next_q_values = self.target_network(next_states_screen, next_states_features).max(1)[0].unsqueeze(1)

            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
            
            # Calcular loss con pesos de importancia
            td_errors = target_q_values - current_q_values
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        # Backward pass
        self.optimizer.zero_grad()
        if autocast_enabled and self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Actualizar prioridades si es necesario
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            errors = td_errors.abs().cpu().detach().numpy().flatten()
            self.memory.update_priorities(indices, errors)

        self.lr_scheduler.step()
        self.training_step += 1
        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Actualizar la red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.info("Red objetivo actualizada.")

    def decay_epsilon(self):
        """Reducir epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        """Guardar modelo entrenado"""
        self.logger.info("Guardando modelo en %s...", path)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }, path)
        self.logger.info("Modelo guardado.")

    def load_model(self, path: str):
        """Cargar modelo entrenado"""
        self.logger.info("Cargando modelo desde %s...", path)
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
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
