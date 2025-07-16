"""Snake RL - Reinforcement Learning implementation for Snake Game.

This package provides Q-Learning and Deep Q-Network implementations
for training AI agents to play the classic Snake game.
"""

__version__ = "0.1.0"

from .environment import SnakeEnvironment
from .agent import TabularQAgent, DQNAgent
from .trainer import SnakeTrainer
from .config import QLearningConfig, DQNConfig, get_default_config, get_preset_config

__all__ = [
    "SnakeEnvironment",
    "TabularQAgent", 
    "DQNAgent",
    "SnakeTrainer",
    "QLearningConfig",
    "DQNConfig",
    "get_default_config",
    "get_preset_config"
] 