"""State representation utilities for Snake RL."""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class StateProcessor:
    """Processes and encodes game states for RL agents."""
    
    def __init__(self, grid_width: int, grid_height: int):
        """Initialize the state processor.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.state_size = grid_width * grid_height
        
        logger.info(f"StateProcessor initialized for {grid_width}x{grid_height} grid")
    
    def encode_grid_state(self, 
                         snake_head: Dict[str, int],
                         snake_body: List[Dict[str, int]], 
                         food_pos: Tuple[int, int],
                         cell_size: int) -> np.ndarray:
        """Encode game state as a grid representation.
        
        Args:
            snake_head: Snake head position {x, y}
            snake_body: List of snake body segments
            food_pos: Food position (x, y)
            cell_size: Size of each cell in pixels
            
        Returns:
            Flattened grid state array
        """
        # Create grid representation
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        # Mark snake body (value = 2)
        for segment in snake_body[1:]:  # Exclude head
            grid_x = segment["x"] // cell_size
            grid_y = segment["y"] // cell_size
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                grid[grid_y, grid_x] = 2
        
        # Mark snake head (value = 1)
        head_grid_x = snake_head["x"] // cell_size
        head_grid_y = snake_head["y"] // cell_size
        if 0 <= head_grid_x < self.grid_width and 0 <= head_grid_y < self.grid_height:
            grid[head_grid_y, head_grid_x] = 1
        
        # Mark food (value = 3)
        food_x, food_y = food_pos
        food_grid_x = food_x // cell_size
        food_grid_y = food_y // cell_size
        if 0 <= food_grid_x < self.grid_width and 0 <= food_grid_y < self.grid_height:
            grid[food_grid_y, food_grid_x] = 3
        
        return grid.flatten()
    
    def encode_feature_state(self,
                           snake_head: Dict[str, int],
                           snake_body: List[Dict[str, int]],
                           food_pos: Tuple[int, int],
                           direction: Tuple[int, int],
                           cell_size: int) -> np.ndarray:
        """Encode game state as engineered features.
        
        Args:
            snake_head: Snake head position
            snake_body: Snake body segments
            food_pos: Food position
            direction: Current movement direction
            cell_size: Size of each cell
            
        Returns:
            Feature vector
        """
        features = []
        
        # Snake head position (normalized)
        head_x = snake_head["x"] / (self.grid_width * cell_size)
        head_y = snake_head["y"] / (self.grid_height * cell_size)
        features.extend([head_x, head_y])
        
        # Food position (normalized)
        food_x, food_y = food_pos
        food_norm_x = food_x / (self.grid_width * cell_size)
        food_norm_y = food_y / (self.grid_height * cell_size)
        features.extend([food_norm_x, food_norm_y])
        
        # Distance to food (normalized)
        distance = np.sqrt((head_x - food_norm_x)**2 + (head_y - food_norm_y)**2)
        features.append(distance)
        
        # Direction to food (normalized)
        if food_norm_x > head_x:
            food_dir_x = 1
        elif food_norm_x < head_x:
            food_dir_x = -1
        else:
            food_dir_x = 0
            
        if food_norm_y > head_y:
            food_dir_y = 1
        elif food_norm_y < head_y:
            food_dir_y = -1
        else:
            food_dir_y = 0
            
        features.extend([food_dir_x, food_dir_y])
        
        # Current direction
        dir_x, dir_y = direction
        dir_norm_x = dir_x / cell_size if dir_x != 0 else 0
        dir_norm_y = dir_y / cell_size if dir_y != 0 else 0
        features.extend([dir_norm_x, dir_norm_y])
        
        # Snake length
        snake_length = len(snake_body) / (self.grid_width * self.grid_height)  # Normalized
        features.append(snake_length)
        
        # Danger detection (binary features for each direction)
        dangers = self._detect_dangers(snake_head, snake_body, cell_size)
        features.extend(dangers)
        
        # Wall distances (normalized)
        wall_distances = self._calculate_wall_distances(snake_head, cell_size)
        features.extend(wall_distances)
        
        return np.array(features, dtype=np.float32)
    
    def _detect_dangers(self, 
                       snake_head: Dict[str, int], 
                       snake_body: List[Dict[str, int]], 
                       cell_size: int) -> List[int]:
        """Detect dangers in each direction.
        
        Returns:
            Binary list [up_danger, down_danger, left_danger, right_danger]
        """
        dangers = []
        directions = [
            (0, -cell_size),   # UP
            (0, cell_size),    # DOWN
            (-cell_size, 0),   # LEFT
            (cell_size, 0)     # RIGHT
        ]
        
        for dx, dy in directions:
            next_pos = {
                "x": snake_head["x"] + dx,
                "y": snake_head["y"] + dy
            }
            
            # Check wall collision
            wall_danger = (next_pos["x"] < 0 or 
                          next_pos["x"] >= self.grid_width * cell_size or
                          next_pos["y"] < 0 or 
                          next_pos["y"] >= self.grid_height * cell_size)
            
            # Check self collision
            self_danger = any(
                segment["x"] == next_pos["x"] and segment["y"] == next_pos["y"]
                for segment in snake_body
            )
            
            dangers.append(1 if (wall_danger or self_danger) else 0)
        
        return dangers
    
    def _calculate_wall_distances(self, 
                                 snake_head: Dict[str, int], 
                                 cell_size: int) -> List[float]:
        """Calculate normalized distances to walls.
        
        Returns:
            List [up_distance, down_distance, left_distance, right_distance]
        """
        head_x = snake_head["x"]
        head_y = snake_head["y"]
        
        max_width = self.grid_width * cell_size
        max_height = self.grid_height * cell_size
        
        distances = [
            head_y / max_height,                    # Distance to top wall
            (max_height - head_y) / max_height,     # Distance to bottom wall
            head_x / max_width,                     # Distance to left wall
            (max_width - head_x) / max_width        # Distance to right wall
        ]
        
        return distances
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values for neural network input.
        
        Args:
            state: Raw state array
            
        Returns:
            Normalized state array
        """
        if state.dtype == np.int8:
            # Grid state: normalize to [0, 1]
            return state.astype(np.float32) / 3.0  # Max value in grid is 3
        else:
            # Feature state: already normalized during creation
            return state.astype(np.float32)
    
    def get_state_hash(self, state: np.ndarray) -> str:
        """Get a hash representation of the state for Q-table indexing.
        
        Args:
            state: State array
            
        Returns:
            String hash of the state
        """
        return str(state.tobytes())


class StateHistory:
    """Maintains a history of states for frame stacking."""
    
    def __init__(self, history_length: int, state_size: int):
        """Initialize state history buffer.
        
        Args:
            history_length: Number of frames to stack
            state_size: Size of each state
        """
        self.history_length = history_length
        self.state_size = state_size
        self.history = np.zeros((history_length, state_size), dtype=np.float32)
        
        logger.info(f"StateHistory initialized with {history_length} frames")
    
    def add_state(self, state: np.ndarray) -> None:
        """Add a new state to the history.
        
        Args:
            state: New state to add
        """
        # Shift history and add new state
        self.history[:-1] = self.history[1:]
        self.history[-1] = state
    
    def get_stacked_state(self) -> np.ndarray:
        """Get the stacked state representation.
        
        Returns:
            Flattened stacked state
        """
        return self.history.flatten()
    
    def reset(self, initial_state: np.ndarray) -> None:
        """Reset history with initial state.
        
        Args:
            initial_state: State to fill the history with
        """
        self.history.fill(0)
        for i in range(self.history_length):
            self.history[i] = initial_state 