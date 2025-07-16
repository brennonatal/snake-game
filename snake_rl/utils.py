"""Utility functions for Snake RL."""

import numpy as np
import random
import os
import json
import pickle
from typing import Any, Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seeds set to {seed} (including PyTorch)")
    except ImportError:
        logger.info(f"Random seeds set to {seed} (PyTorch not available)")


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Created directory: {directory}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON data to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON data from {filepath}")
    return data


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.debug(f"Saved pickle data to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.debug(f"Loaded pickle data from {filepath}")
    return data


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int], metric: str = "manhattan") -> float:
    """Calculate distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        metric: Distance metric ("manhattan", "euclidean")
        
    Returns:
        Distance value
    """
    x1, y1 = pos1
    x2, y2 = pos2
    
    if metric == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)
    elif metric == "euclidean":
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def normalize_position(pos: Tuple[int, int], grid_size: Tuple[int, int]) -> Tuple[float, float]:
    """Normalize position to [0, 1] range.
    
    Args:
        pos: Position (x, y)
        grid_size: Grid dimensions (width, height)
        
    Returns:
        Normalized position
    """
    x, y = pos
    width, height = grid_size
    return x / width, y / height


def get_direction_vector(direction: str) -> Tuple[int, int]:
    """Get direction vector from string.
    
    Args:
        direction: Direction string ("UP", "DOWN", "LEFT", "RIGHT")
        
    Returns:
        Direction vector (dx, dy)
    """
    directions = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }
    return directions.get(direction.upper(), (0, 0))


def get_opposite_direction(direction: str) -> str:
    """Get opposite direction.
    
    Args:
        direction: Current direction
        
    Returns:
        Opposite direction
    """
    opposites = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT"
    }
    return opposites.get(direction.upper(), direction)


def is_valid_action(current_direction: Tuple[int, int], new_direction: Tuple[int, int]) -> bool:
    """Check if new direction is valid (not opposite to current).
    
    Args:
        current_direction: Current movement direction
        new_direction: Proposed new direction
        
    Returns:
        True if valid action
    """
    # If not moving, any direction is valid
    if current_direction == (0, 0):
        return True
    
    # Check if new direction is opposite
    return new_direction != (-current_direction[0], -current_direction[1])


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_timestamp() -> str:
    """Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average of data.
    
    Args:
        data: List of values
        window_size: Size of moving window
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return [np.mean(data[:i+1]) for i in range(len(data))]
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        result.append(np.mean(data[start_idx:i+1]))
    
    return result


def exponential_decay(initial_value: float, decay_rate: float, step: int, min_value: float = 0.0) -> float:
    """Calculate exponentially decayed value.
    
    Args:
        initial_value: Initial value
        decay_rate: Decay rate per step
        step: Current step
        min_value: Minimum value
        
    Returns:
        Decayed value
    """
    value = initial_value * (decay_rate ** step)
    return max(value, min_value)


def linear_decay(initial_value: float, final_value: float, current_step: int, total_steps: int) -> float:
    """Calculate linearly decayed value.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        current_step: Current step
        total_steps: Total number of steps
        
    Returns:
        Linearly interpolated value
    """
    if current_step >= total_steps:
        return final_value
    
    progress = current_step / total_steps
    return initial_value + (final_value - initial_value) * progress


def create_grid_from_positions(snake_positions: List[Tuple[int, int]], 
                              food_position: Tuple[int, int],
                              grid_size: Tuple[int, int]) -> np.ndarray:
    """Create grid representation from positions.
    
    Args:
        snake_positions: List of snake segment positions (head first)
        food_position: Food position
        grid_size: Grid dimensions (width, height)
        
    Returns:
        Grid array with encoded positions
    """
    width, height = grid_size
    grid = np.zeros((height, width), dtype=np.int8)
    
    # Mark snake body (excluding head)
    for i, (x, y) in enumerate(snake_positions[1:]):
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 2  # Body
    
    # Mark snake head
    if snake_positions:
        head_x, head_y = snake_positions[0]
        if 0 <= head_x < width and 0 <= head_y < height:
            grid[head_y, head_x] = 1  # Head
    
    # Mark food
    food_x, food_y = food_position
    if 0 <= food_x < width and 0 <= food_y < height:
        grid[food_y, food_x] = 3  # Food
    
    return grid


def get_free_positions(snake_positions: List[Tuple[int, int]], 
                      grid_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get list of free positions on the grid.
    
    Args:
        snake_positions: List of snake segment positions
        grid_size: Grid dimensions (width, height)
        
    Returns:
        List of free positions
    """
    width, height = grid_size
    occupied = set(snake_positions)
    
    free_positions = []
    for x in range(width):
        for y in range(height):
            if (x, y) not in occupied:
                free_positions.append((x, y))
    
    return free_positions


def calculate_reward_components(old_state: Dict[str, Any], 
                               new_state: Dict[str, Any], 
                               action: int,
                               game_status: str) -> Dict[str, float]:
    """Calculate individual reward components.
    
    Args:
        old_state: Previous game state
        new_state: Current game state
        action: Action taken
        game_status: Game status ("normal", "food_eaten", "collision", etc.)
        
    Returns:
        Dictionary of reward components
    """
    rewards = {
        'food_reward': 0.0,
        'collision_penalty': 0.0,
        'distance_reward': 0.0,
        'survival_reward': 0.0,
        'length_reward': 0.0
    }
    
    # Food reward
    if game_status == "food_eaten":
        rewards['food_reward'] = 100.0
    
    # Collision penalty
    if "collision" in game_status:
        rewards['collision_penalty'] = -100.0
    
    # Distance-based reward (if positions available)
    if 'head_pos' in old_state and 'head_pos' in new_state and 'food_pos' in new_state:
        old_distance = calculate_distance(old_state['head_pos'], new_state['food_pos'])
        new_distance = calculate_distance(new_state['head_pos'], new_state['food_pos'])
        
        if new_distance < old_distance:
            rewards['distance_reward'] = 5.0
        elif new_distance > old_distance:
            rewards['distance_reward'] = -2.0
    
    # Survival reward
    if game_status == "normal":
        rewards['survival_reward'] = 1.0
    
    # Length reward
    if 'snake_length' in new_state:
        rewards['length_reward'] = new_state['snake_length'] * 2.0
    
    return rewards


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = datetime.now()
        if self.start_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        if name not in self.timers:
            self.timers[name] = Timer()
        self.timers[name].start()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name in self.timers:
            return self.timers[name].stop()
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all performance statistics."""
        return {
            'counters': self.counters.copy(),
            'timer_names': list(self.timers.keys())
        } 