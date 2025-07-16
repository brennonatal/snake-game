"""Agent implementations for Snake RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import defaultdict, deque
from typing import Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TabularQAgent:
    """Tabular Q-Learning agent for Snake game."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 4,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """Initialize the Tabular Q-Learning agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for Q-learning updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table as defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Statistics
        self.total_steps = 0
        
        logger.info(f"TabularQAgent initialized with lr={learning_rate}, gamma={discount_factor}, epsilon={epsilon}")
    
    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        state_key = self._state_to_key(state)
        
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.randrange(self.action_size)
            logger.debug(f"Random action: {action}")
        else:
            # Exploitation: best action according to Q-table
            q_values = self.q_table[state_key]
            action = np.argmax(q_values)
            logger.debug(f"Greedy action: {action}, Q-values: {q_values}")
        
        return action
    
    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float, 
               next_state: np.ndarray, 
               done: bool) -> None:
        """Update Q-table using Q-learning rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.q_table[next_state_key]
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # Q-learning update
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Update statistics
        self.total_steps += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        logger.debug(f"Q-update: state={state_key[:10]}..., action={action}, "
                    f"reward={reward:.2f}, target_q={target_q:.2f}, current_q={current_q:.2f}")
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state array to string key for Q-table."""
        return str(state.tobytes())
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        logger.info(f"Q-table saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        import pickle
        with open(filepath, 'rb') as f:
            q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(q_dict)
        logger.info(f"Q-table loaded from {filepath}")


class QNetwork(nn.Module):
    """Neural network for Deep Q-Learning."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list = [512, 256, 128]):
        """Initialize the Q-Network.
        
        Args:
            state_size: Input state dimension
            action_size: Output action dimension
            hidden_layers: List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for Snake game."""
    
    def __init__(self,
                 state_size: int,
                 action_size: int = 4,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 device: str = "auto"):
        """Initialize the DQN agent.
        
        Args:
            state_size: Input state dimension
            action_size: Number of possible actions
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Frequency of target network updates
            device: Device to run computations on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Neural networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics
        self.total_steps = 0
        self.loss_history = []
        
        logger.info(f"DQNAgent initialized on device {self.device}")
    
    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy with neural network.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: neural network prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> None:
        """Store experience and train if ready.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if we have enough experiences
        if len(self.replay_buffer) >= self.batch_size:
            self._train()
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug("Target network updated")
        
        self.total_steps += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train(self) -> None:
        """Train the neural network on a batch of experiences."""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
        
        logger.debug(f"Training loss: {loss.item():.4f}")
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"Model loaded from {filepath}") 