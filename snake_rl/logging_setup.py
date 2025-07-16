"""Logging setup for Snake RL training."""

import logging
import logging.handlers
import os
from typing import Optional
from .config import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup comprehensive logging system for Snake RL.
    
    Args:
        config: Logging configuration object
    """
    if config is None:
        config = LoggingConfig()
    
    # Ensure log directories exist
    for log_file in [config.main_log_file, config.agent_log_file, config.environment_log_file]:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=config.log_format,
        datefmt=config.date_format
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.console_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Main log file handler with rotation
    main_file_handler = logging.handlers.RotatingFileHandler(
        config.main_log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    main_file_handler.setLevel(getattr(logging, config.file_level.upper()))
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)
    
    # Setup specific loggers
    _setup_component_logger('snake_rl.agent', config.agent_log_file, config, formatter)
    _setup_component_logger('snake_rl.environment', config.environment_log_file, config, formatter)
    _setup_component_logger('snake_rl.trainer', config.main_log_file, config, formatter)
    
    # Log initial setup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Console level: {config.console_level}")
    logger.info(f"File level: {config.file_level}")
    logger.info(f"Main log file: {config.main_log_file}")


def _setup_component_logger(logger_name: str, 
                           log_file: str, 
                           config: LoggingConfig, 
                           formatter: logging.Formatter) -> None:
    """Setup logger for a specific component.
    
    Args:
        logger_name: Name of the logger
        log_file: Log file path
        config: Logging configuration
        formatter: Log formatter
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    # Console handler for this component
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.console_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for this component
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    file_handler.setLevel(getattr(logging, config.file_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance metrics and timing information."""
    
    def __init__(self, name: str = "performance"):
        """Initialize performance logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"snake_rl.{name}")
        self.timers = {}
    
    def log_hyperparameters(self, params: dict) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_episode_metrics(self, 
                           episode: int, 
                           reward: float, 
                           score: int, 
                           length: int, 
                           epsilon: float,
                           episode_time: float) -> None:
        """Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            score: Episode score (food eaten)
            length: Episode length in steps
            epsilon: Current exploration rate
            episode_time: Time taken for episode
        """
        self.logger.info(
            f"Episode {episode:5d} | Reward: {reward:7.2f} | Score: {score:3d} | "
            f"Length: {length:4d} | Epsilon: {epsilon:.3f} | Time: {episode_time:.3f}s"
        )
    
    def log_evaluation_results(self, episode: int, results: dict) -> None:
        """Log evaluation results.
        
        Args:
            episode: Episode number when evaluation was performed
            results: Dictionary of evaluation results
        """
        self.logger.info(f"Evaluation at episode {episode}:")
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.3f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_training_summary(self, summary: dict) -> None:
        """Log training summary.
        
        Args:
            summary: Dictionary of training summary metrics
        """
        self.logger.info("Training Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.3f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def start_timer(self, name: str) -> None:
        """Start a named timer.
        
        Args:
            name: Timer name
        """
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and log the result.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        import time
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            self.logger.debug(f"Timer '{name}': {elapsed:.3f}s")
            del self.timers[name]
            return elapsed
        return 0.0


class MemoryLogger:
    """Logger for memory usage monitoring."""
    
    def __init__(self, name: str = "memory"):
        """Initialize memory logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"snake_rl.{name}")
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage.
        
        Args:
            context: Context description for the memory log
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            context_str = f" ({context})" if context else ""
            self.logger.debug(f"Memory usage{context_str}: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
    
    def log_gpu_memory(self) -> None:
        """Log GPU memory usage if available."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                self.logger.debug(f"GPU memory - Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")
        except ImportError:
            pass


def log_system_info() -> None:
    """Log system information."""
    logger = logging.getLogger("snake_rl.system")
    
    # Python version
    import sys
    logger.info(f"Python version: {sys.version}")
    
    # Platform information
    import platform
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # PyTorch information
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    # NumPy version
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        logger.warning("NumPy not available")


# Context manager for timing operations
class LoggedTimer:
    """Context manager for timing operations with logging."""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        """Initialize logged timer.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.debug(f"Completed {self.operation_name} in {elapsed:.3f}s")


# Decorator for logging function calls
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls.
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator 