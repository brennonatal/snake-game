"""State representation utilities for Snake RL."""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class StateProcessor:
    """Processes and encodes game states for RL agents using enhanced feature vectors."""
    
    def __init__(self, grid_width: int, grid_height: int):
        """Initialize the state processor.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        # Enhanced feature vector: 24 features instead of 2400
        self.state_size = 24
        
        logger.info(f"StateProcessor initialized for {grid_width}x{grid_height} grid with {self.state_size} features")
    
    def encode_state(self, 
                    snake_head: Dict[str, int],
                    snake_body: List[Dict[str, int]], 
                    food_pos: Tuple[int, int],
                    direction: Tuple[int, int],
                    cell_size: int) -> np.ndarray:
        """Encode game state as an enhanced feature vector.
        
        Args:
            snake_head: Snake head position {x, y}
            snake_body: List of snake body segments
            food_pos: Food position (x, y)
            direction: Current movement direction (dx, dy)
            cell_size: Size of each cell in pixels
            
        Returns:
            24-dimensional feature vector
        """
        features = []
        
        # Convert positions to grid coordinates
        head_grid_x = snake_head["x"] // cell_size
        head_grid_y = snake_head["y"] // cell_size
        food_grid_x, food_grid_y = food_pos[0] // cell_size, food_pos[1] // cell_size
        
        # 1-2: Snake head position (normalized)
        head_norm_x = head_grid_x / self.grid_width
        head_norm_y = head_grid_y / self.grid_height
        features.extend([head_norm_x, head_norm_y])
        
        # 3-4: Food position (normalized)
        food_norm_x = food_grid_x / self.grid_width
        food_norm_y = food_grid_y / self.grid_height
        features.extend([food_norm_x, food_norm_y])
        
        # 5-6: Relative food direction (normalized)
        rel_x = (food_grid_x - head_grid_x) / self.grid_width
        rel_y = (food_grid_y - head_grid_y) / self.grid_height
        features.extend([rel_x, rel_y])
        
        # 7: Distance to food (normalized)
        distance = np.sqrt(rel_x**2 + rel_y**2)
        features.append(distance)
        
        # 8-9: Current movement direction (normalized)
        dir_x = direction[0] / cell_size if direction[0] != 0 else 0
        dir_y = direction[1] / cell_size if direction[1] != 0 else 0
        features.extend([dir_x, dir_y])
        
        # 10: Snake length (normalized)
        snake_length = len(snake_body) / (self.grid_width * self.grid_height)
        features.append(snake_length)
        
        # 11-14: Immediate danger detection (binary)
        dangers = self._detect_immediate_dangers(snake_head, snake_body, cell_size)
        features.extend(dangers)
        
        # 15-18: Wall distances (normalized)
        wall_distances = self._calculate_wall_distances(head_grid_x, head_grid_y)
        features.extend(wall_distances)
        
        # 19-22: Body proximity in each direction (normalized)
        body_proximity = self._calculate_body_proximity(head_grid_x, head_grid_y, snake_body, cell_size)
        features.extend(body_proximity)
        
        # 23: Movement alignment with food direction
        alignment = self._calculate_movement_alignment(direction, rel_x, rel_y, cell_size)
        features.append(alignment)
        
        # 24: Food reachability heuristic
        reachability = self._calculate_food_reachability(head_grid_x, head_grid_y, food_grid_x, food_grid_y, snake_body, cell_size)
        features.append(reachability)
        
        return np.array(features, dtype=np.float32)
    
    def _detect_immediate_dangers(self, 
                                 snake_head: Dict[str, int], 
                                 snake_body: List[Dict[str, int]], 
                                 cell_size: int) -> List[int]:
        """Detect immediate dangers in each direction.
        
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
            next_x = snake_head["x"] + dx
            next_y = snake_head["y"] + dy
            
            # Check wall collision
            wall_danger = (next_x < 0 or 
                          next_x >= self.grid_width * cell_size or
                          next_y < 0 or 
                          next_y >= self.grid_height * cell_size)
            
            # Check self collision
            self_danger = any(
                segment["x"] == next_x and segment["y"] == next_y
                for segment in snake_body
            )
            
            dangers.append(1 if (wall_danger or self_danger) else 0)
        
        return dangers
    
    def _calculate_wall_distances(self, head_grid_x: int, head_grid_y: int) -> List[float]:
        """Calculate normalized distances to walls.
        
        Returns:
            List [up_distance, down_distance, left_distance, right_distance]
        """
        distances = [
            head_grid_y / self.grid_height,                           # Distance to top wall
            (self.grid_height - 1 - head_grid_y) / self.grid_height,  # Distance to bottom wall
            head_grid_x / self.grid_width,                            # Distance to left wall
            (self.grid_width - 1 - head_grid_x) / self.grid_width     # Distance to right wall
        ]
        
        return distances
    
    def _calculate_body_proximity(self, 
                                 head_grid_x: int, 
                                 head_grid_y: int, 
                                 snake_body: List[Dict[str, int]], 
                                 cell_size: int) -> List[float]:
        """Calculate closest body segment distance in each direction.
        
        Returns:
            List [up_proximity, down_proximity, left_proximity, right_proximity]
        """
        proximities = [1.0, 1.0, 1.0, 1.0]  # Initialize to max distance
        max_distance = max(self.grid_width, self.grid_height)
        
        for segment in snake_body[1:]:  # Skip head
            seg_grid_x = segment["x"] // cell_size
            seg_grid_y = segment["y"] // cell_size
            
            # Up direction
            if seg_grid_x == head_grid_x and seg_grid_y < head_grid_y:
                distance = (head_grid_y - seg_grid_y) / max_distance
                proximities[0] = min(proximities[0], distance)
            
            # Down direction
            if seg_grid_x == head_grid_x and seg_grid_y > head_grid_y:
                distance = (seg_grid_y - head_grid_y) / max_distance
                proximities[1] = min(proximities[1], distance)
            
            # Left direction
            if seg_grid_y == head_grid_y and seg_grid_x < head_grid_x:
                distance = (head_grid_x - seg_grid_x) / max_distance
                proximities[2] = min(proximities[2], distance)
            
            # Right direction
            if seg_grid_y == head_grid_y and seg_grid_x > head_grid_x:
                distance = (seg_grid_x - head_grid_x) / max_distance
                proximities[3] = min(proximities[3], distance)
        
        return proximities
    
    def _calculate_movement_alignment(self, 
                                    direction: Tuple[int, int], 
                                    rel_x: float, 
                                    rel_y: float, 
                                    cell_size: int) -> float:
        """Calculate how well current movement aligns with food direction.
        
        Returns:
            Alignment score between -1 (opposite) and 1 (same direction)
        """
        if direction[0] == 0 and direction[1] == 0:
            return 0.0
        
        # Normalize direction vector
        dir_magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
        norm_dir_x = direction[0] / dir_magnitude
        norm_dir_y = direction[1] / dir_magnitude
        
        # Normalize food direction vector
        food_magnitude = np.sqrt(rel_x**2 + rel_y**2)
        if food_magnitude == 0:
            return 0.0
        
        norm_food_x = rel_x / food_magnitude
        norm_food_y = rel_y / food_magnitude
        
        # Calculate dot product (cosine similarity)
        alignment = norm_dir_x * norm_food_x + norm_dir_y * norm_food_y
        
        return alignment
    
    def _calculate_food_reachability(self, 
                                   head_grid_x: int, 
                                   head_grid_y: int, 
                                   food_grid_x: int, 
                                   food_grid_y: int, 
                                   snake_body: List[Dict[str, int]], 
                                   cell_size: int) -> float:
        """Calculate a heuristic for how easily the food can be reached.
        
        Returns:
            Reachability score between 0 (blocked) and 1 (clear path)
        """
        # Manhattan distance to food
        manhattan_dist = abs(food_grid_x - head_grid_x) + abs(food_grid_y - head_grid_y)
        if manhattan_dist == 0:
            return 1.0
        
        # Check if there are body segments blocking the direct path
        blocked_cells = 0
        total_path_cells = manhattan_dist
        
        # Simple line-of-sight check (not perfect but good heuristic)
        dx = 1 if food_grid_x > head_grid_x else -1 if food_grid_x < head_grid_x else 0
        dy = 1 if food_grid_y > head_grid_y else -1 if food_grid_y < head_grid_y else 0
        
        # Check horizontal path
        x = head_grid_x
        while x != food_grid_x:
            x += dx
            if any(segment["x"] // cell_size == x and segment["y"] // cell_size == head_grid_y 
                   for segment in snake_body[1:]):
                blocked_cells += 1
        
        # Check vertical path
        y = head_grid_y
        while y != food_grid_y:
            y += dy
            if any(segment["x"] // cell_size == food_grid_x and segment["y"] // cell_size == y 
                   for segment in snake_body[1:]):
                blocked_cells += 1
        
        # Calculate reachability (higher when fewer blocks)
        if total_path_cells == 0:
            return 1.0
        
        reachability = 1.0 - (blocked_cells / total_path_cells)
        return max(0.0, reachability)
    
    # Keep legacy method for backward compatibility but redirect to new method
    def encode_grid_state(self, 
                         snake_head: Dict[str, int],
                         snake_body: List[Dict[str, int]], 
                         food_pos: Tuple[int, int],
                         cell_size: int) -> np.ndarray:
        """Legacy method - redirects to feature-based encoding.
        
        Note: This method is deprecated. Use encode_state() instead.
        """
        logger.warning("encode_grid_state() is deprecated. Use encode_state() instead.")
        # Default direction if not provided
        direction = (0, cell_size)  # Default to DOWN
        return self.encode_state(snake_head, snake_body, food_pos, direction, cell_size)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values for neural network input.
        
        Args:
            state: Raw state array
            
        Returns:
            Normalized state array (features are already normalized during encoding)
        """
        return state.astype(np.float32)
    
    def get_state_hash(self, state: np.ndarray) -> str:
        """Get a hash representation of the state for Q-table indexing.
        
        Args:
            state: State array
            
        Returns:
            String hash of the state
        """
        # Round to reduce state space for tabular methods
        rounded_state = np.round(state, decimals=2)
        return str(rounded_state.tobytes())


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
        
        logger.info(f"StateHistory initialized with {history_length} frames of size {state_size}")
    
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