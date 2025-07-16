"""Configuration classes for Snake RL training."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class QLearningConfig:
    """Configuration for Tabular Q-Learning."""
    
    # Learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Environment parameters
    grid_width: int = 60
    grid_height: int = 40
    cell_size: int = 10
    
    # Training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    
    # Evaluation parameters
    eval_frequency: int = 100
    eval_episodes: int = 10
    
    # Logging parameters
    log_frequency: int = 50
    metrics_window: int = 100
    
    # Checkpointing parameters
    save_frequency: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Success criteria
    target_score: float = 20.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'cell_size': self.cell_size,
            'max_episodes': self.max_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'log_frequency': self.log_frequency,
            'metrics_window': self.metrics_window,
            'save_frequency': self.save_frequency,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'target_score': self.target_score
        }


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network."""
    
    # Network architecture
    hidden_layers: List[int] = None
    learning_rate: float = 0.001
    
    # Training parameters
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 32
    min_replay_size: int = 1000
    
    # Target network
    target_update_frequency: int = 1000
    
    # Environment parameters
    grid_width: int = 60
    grid_height: int = 40
    cell_size: int = 10
    
    # Training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    train_frequency: int = 4
    
    # Evaluation parameters
    eval_frequency: int = 100
    eval_episodes: int = 10
    
    # Logging parameters
    log_frequency: int = 50
    metrics_window: int = 100
    
    # Checkpointing parameters
    save_frequency: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device parameters
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Success criteria
    target_score: float = 20.0
    
    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'min_replay_size': self.min_replay_size,
            'target_update_frequency': self.target_update_frequency,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'cell_size': self.cell_size,
            'max_episodes': self.max_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'train_frequency': self.train_frequency,
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'log_frequency': self.log_frequency,
            'metrics_window': self.metrics_window,
            'save_frequency': self.save_frequency,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'target_score': self.target_score
        }


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    
    # Logging levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log file paths
    main_log_file: str = "logs/training.log"
    agent_log_file: str = "logs/agent.log"
    environment_log_file: str = "logs/environment.log"
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Performance logging
    log_performance: bool = True
    log_hyperparameters: bool = True
    log_model_updates: bool = False  # Set to True for detailed debugging
    
    # Log rotation
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'console_level': self.console_level,
            'file_level': self.file_level,
            'main_log_file': self.main_log_file,
            'agent_log_file': self.agent_log_file,
            'environment_log_file': self.environment_log_file,
            'log_format': self.log_format,
            'date_format': self.date_format,
            'log_performance': self.log_performance,
            'log_hyperparameters': self.log_hyperparameters,
            'log_model_updates': self.log_model_updates,
            'max_file_size': self.max_file_size,
            'backup_count': self.backup_count
        }


def get_default_config(algorithm: str = "tabular") -> QLearningConfig | DQNConfig:
    """Get default configuration for specified algorithm.
    
    Args:
        algorithm: Algorithm type ("tabular" or "dqn")
        
    Returns:
        Default configuration object
        
    Raises:
        ValueError: If algorithm is not supported
    """
    if algorithm.lower() == "tabular":
        return QLearningConfig()
    elif algorithm.lower() == "dqn":
        return DQNConfig()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'tabular' or 'dqn'.")


def load_config_from_dict(config_dict: Dict[str, Any], algorithm: str) -> QLearningConfig | DQNConfig:
    """Load configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        algorithm: Algorithm type ("tabular" or "dqn")
        
    Returns:
        Configuration object
    """
    if algorithm.lower() == "tabular":
        return QLearningConfig(**config_dict)
    elif algorithm.lower() == "dqn":
        return DQNConfig(**config_dict)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


# Pre-defined configuration presets
PRESETS = {
    "tabular_fast": QLearningConfig(
        learning_rate=0.2,
        epsilon_decay=0.99,
        max_episodes=5000,
        log_frequency=25
    ),
    
    "tabular_stable": QLearningConfig(
        learning_rate=0.05,
        epsilon_decay=0.999,
        max_episodes=15000,
        target_score=25.0
    ),
    
    "dqn_fast": DQNConfig(
        hidden_layers=[256, 128],
        learning_rate=0.002,
        epsilon_decay=0.99,
        max_episodes=5000,
        buffer_size=5000,
        target_update_frequency=500
    ),
    
    "dqn_stable": DQNConfig(
        hidden_layers=[512, 256, 128],
        learning_rate=0.0005,
        epsilon_decay=0.999,
        max_episodes=15000,
        buffer_size=20000,
        target_update_frequency=2000,
        target_score=25.0
    )
}


def get_preset_config(preset_name: str) -> QLearningConfig | DQNConfig:
    """Get predefined configuration preset.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        Configuration object
        
    Raises:
        ValueError: If preset is not found
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return PRESETS[preset_name] 