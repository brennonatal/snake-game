"""Game engine for Snake game - extracted from main.py for reusability.

This module contains the core game logic separated from rendering,
making it suitable for both human play and RL training.
"""

import random
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class GameStatus(Enum):
    """Game status enumeration."""
    PLAYING = "playing"
    GAME_OVER = "game_over"
    FOOD_EATEN = "food_eaten"
    WALL_COLLISION = "wall_collision"
    SELF_COLLISION = "self_collision"
    MAX_STEPS = "max_steps"


@dataclass
class SnakeCell:
    """Represents a snake cell with position and size."""
    x: int = 300
    y: int = 200
    size: int = 10
    
    def reset(self) -> None:
        """Reset cell to default position."""
        self.x = 300
        self.y = 200


@dataclass
class GameState:
    """Represents the complete game state."""
    snake_head: SnakeCell
    snake_body: List[SnakeCell]
    food_x: int
    food_y: int
    score: int
    x_move: int
    y_move: int
    game_over: bool
    steps: int
    status: GameStatus


class SnakeGameEngine:
    """Core game engine for Snake game."""
    
    def __init__(self, 
                 width: int = 600, 
                 height: int = 400,
                 cell_size: int = 10,
                 food_ratio: int = 4,
                 initial_speed: int = 10,
                 max_steps: Optional[int] = None):
        """Initialize the game engine.
        
        Args:
            width: Game board width in pixels
            height: Game board height in pixels
            cell_size: Size of each cell in pixels
            food_ratio: Radius of food circle
            initial_speed: Initial snake speed
            max_steps: Maximum steps per episode (None for unlimited)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.food_ratio = food_ratio
        self.initial_speed = initial_speed
        self.max_steps = max_steps
        
        # Initialize game state
        self.reset()
    
    def reset(self) -> GameState:
        """Reset the game to initial state.
        
        Returns:
            Initial game state
        """
        # Initialize snake
        self.snake_head = SnakeCell()
        self.snake_body = [self.snake_head]
        self.snake_speed = self.initial_speed
        
        # Initialize movement
        self.x_move = 0
        self.y_move = 0
        
        # Initialize food
        self._place_food()
        
        # Initialize game state
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.status = GameStatus.PLAYING
        
        return self.get_state()
    
    def step(self, action: Optional[str] = None) -> Tuple[GameState, Dict[str, Any]]:
        """Execute one game step.
        
        Args:
            action: Action to take ("UP", "DOWN", "LEFT", "RIGHT", or None for continue current direction)
            
        Returns:
            Tuple of (new_game_state, info_dict)
        """
        if self.game_over:
            return self.get_state(), {"status": self.status.value, "message": "Game already over"}
        
        # Update movement based on action
        if action:
            self._update_movement(action)
        
        # Move snake
        old_head_pos = (self.snake_head.x, self.snake_head.y)
        new_head = SnakeCell(
            x=self.snake_head.x + self.x_move,
            y=self.snake_head.y + self.y_move,
            size=self.cell_size
        )
        
        # Check collisions before updating state
        collision_info = self._check_collisions(new_head)
        if collision_info["collision"]:
            self.game_over = True
            self.status = GameStatus.WALL_COLLISION if collision_info["type"] == "wall" else GameStatus.SELF_COLLISION
            return self.get_state(), {
                "status": self.status.value,
                "collision_type": collision_info["type"],
                "position": (new_head.x, new_head.y)
            }
        
        # Update snake position
        self.snake_head = new_head
        self.snake_body.insert(0, self.snake_head)
        
        # Check food consumption
        food_eaten = self._check_food_eaten()
        if food_eaten:
            self.score += 1
            self.status = GameStatus.FOOD_EATEN
            self._place_food()
            # Increase speed slightly
            self.snake_speed += 0.5
            info = {
                "status": self.status.value,
                "score": self.score,
                "snake_length": len(self.snake_body)
            }
        else:
            # Remove tail if no food eaten
            self.snake_body.pop()
            self.status = GameStatus.PLAYING
            info = {
                "status": self.status.value,
                "score": self.score,
                "snake_length": len(self.snake_body)
            }
        
        # Update step counter
        self.steps += 1
        
        # Check for max steps
        if self.max_steps and self.steps >= self.max_steps:
            self.game_over = True
            self.status = GameStatus.MAX_STEPS
            info["status"] = self.status.value
            info["message"] = f"Maximum steps ({self.max_steps}) reached"
        
        return self.get_state(), info
    
    def get_state(self) -> GameState:
        """Get current game state.
        
        Returns:
            Current game state
        """
        return GameState(
            snake_head=self.snake_head,
            snake_body=self.snake_body.copy(),
            food_x=self.food_x,
            food_y=self.food_y,
            score=self.score,
            x_move=self.x_move,
            y_move=self.y_move,
            game_over=self.game_over,
            steps=self.steps,
            status=self.status
        )
    
    def _update_movement(self, action: str) -> None:
        """Update movement direction based on action.
        
        Args:
            action: Direction action ("UP", "DOWN", "LEFT", "RIGHT")
        """
        action = action.upper()
        
        # Prevent immediate death by moving in opposite direction
        if action == "LEFT" and self.x_move == 0:
            self.x_move = -self.cell_size
            self.y_move = 0
        elif action == "RIGHT" and self.x_move == 0:
            self.x_move = self.cell_size
            self.y_move = 0
        elif action == "UP" and self.y_move == 0:
            self.y_move = -self.cell_size
            self.x_move = 0
        elif action == "DOWN" and self.y_move == 0:
            self.y_move = self.cell_size
            self.x_move = 0
    
    def _check_collisions(self, new_head: SnakeCell) -> Dict[str, Any]:
        """Check for collisions with walls or self.
        
        Args:
            new_head: New head position to check
            
        Returns:
            Dictionary with collision information
        """
        # Check wall collision
        if (new_head.x >= self.width or new_head.x < 0 or 
            new_head.y >= self.height or new_head.y < 0):
            return {"collision": True, "type": "wall"}
        
        # Check self collision
        for cell in self.snake_body:
            if (new_head.x < cell.x + cell.size and 
                new_head.x + new_head.size > cell.x and 
                new_head.y < cell.y + cell.size and 
                new_head.y + new_head.size > cell.y):
                return {"collision": True, "type": "self"}
        
        return {"collision": False, "type": None}
    
    def _check_food_eaten(self) -> bool:
        """Check if snake head is at food position.
        
        Returns:
            True if food was eaten
        """
        return ate_food(
            food_x=self.food_x,
            food_y=self.food_y,
            food_ratio=self.food_ratio,
            snake_x=self.snake_head.x,
            snake_y=self.snake_head.y,
            snake_size=self.snake_head.size
        )
    
    def _place_food(self) -> None:
        """Place food at a random location not occupied by snake."""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            self.food_x = round(random.randrange(10, self.width - 10) / 10) * 10
            self.food_y = round(random.randrange(10, self.height - 10) / 10) * 10
            
            # Check if food position conflicts with snake
            food_conflicts = False
            for cell in self.snake_body:
                if (abs(self.food_x - cell.x) < self.cell_size and 
                    abs(self.food_y - cell.y) < self.cell_size):
                    food_conflicts = True
                    break
            
            if not food_conflicts:
                break
            
            attempts += 1
        
        # If we couldn't find a free position, place it randomly anyway
        if attempts >= max_attempts:
            self.food_x = round(random.randrange(10, self.width - 10) / 10) * 10
            self.food_y = round(random.randrange(10, self.height - 10) / 10) * 10
    
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions from current state.
        
        Returns:
            List of valid action strings
        """
        valid_actions = []
        
        if self.x_move == 0:  # Can move left or right
            valid_actions.extend(["LEFT", "RIGHT"])
        if self.y_move == 0:  # Can move up or down
            valid_actions.extend(["UP", "DOWN"])
        
        # If not moving, all directions are valid
        if self.x_move == 0 and self.y_move == 0:
            valid_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        return valid_actions
    
    def get_snake_positions(self) -> List[Tuple[int, int]]:
        """Get snake segment positions as list of tuples.
        
        Returns:
            List of (x, y) positions for each snake segment
        """
        return [(cell.x, cell.y) for cell in self.snake_body]
    
    def get_food_position(self) -> Tuple[int, int]:
        """Get food position.
        
        Returns:
            Food position as (x, y) tuple
        """
        return (self.food_x, self.food_y)
    
    def get_game_metrics(self) -> Dict[str, Any]:
        """Get current game metrics.
        
        Returns:
            Dictionary of game metrics
        """
        return {
            "score": self.score,
            "snake_length": len(self.snake_body),
            "steps": self.steps,
            "food_position": self.get_food_position(),
            "snake_positions": self.get_snake_positions(),
            "current_direction": (self.x_move, self.y_move),
            "game_over": self.game_over,
            "status": self.status.value
        }


# Utility functions (extracted from util.py for consistency)
def ate_food(food_x: int, food_y: int, food_ratio: int, 
             snake_x: int, snake_y: int, snake_size: int = 10) -> bool:
    """Check if snake hit the food.
    
    Args:
        food_x: Food x coordinate
        food_y: Food y coordinate
        food_ratio: Food radius
        snake_x: Snake head x coordinate
        snake_y: Snake head y coordinate
        snake_size: Snake head size
        
    Returns:
        True if food was eaten
    """
    # Auxiliary variables to calculate distance
    test_x = food_x
    test_y = food_y

    # Check which is the closest axis
    if food_x < snake_x:
        test_x = snake_x    
    elif food_x > snake_x + snake_size:
        test_x = snake_x + snake_size 
    if food_y < snake_y:
        test_y = snake_y    
    elif food_y > snake_y + snake_size:
        test_y = snake_y + snake_size 

    # Get distance from closest edges
    dist_x = food_x - test_x
    dist_y = food_y - test_y
    distance = math.sqrt((dist_x * dist_x) + (dist_y * dist_y))

    # If the distance is less than the food_ratio, the snake ate the food
    return distance <= food_ratio


def snake_collision(snake_head: SnakeCell, snake_body: List[SnakeCell]) -> bool:
    """Check if snake head collides with body.
    
    Args:
        snake_head: Snake head cell
        snake_body: List of snake body cells
        
    Returns:
        True if collision detected
    """
    for cell in snake_body:
        if (snake_head.x < cell.x + cell.size and 
            snake_head.x + snake_head.size > cell.x and 
            snake_head.y < cell.y + cell.size and 
            snake_head.y + snake_head.size > cell.y):
            return True
    return False


# Factory function for easy game creation
def create_game(width: int = 600, 
               height: int = 400, 
               cell_size: int = 10,
               headless: bool = False,
               max_steps: Optional[int] = None) -> SnakeGameEngine:
    """Create a new Snake game instance.
    
    Args:
        width: Game board width
        height: Game board height
        cell_size: Size of each cell
        headless: Whether to run in headless mode (for RL)
        max_steps: Maximum steps per episode
        
    Returns:
        New game engine instance
    """
    return SnakeGameEngine(
        width=width,
        height=height,
        cell_size=cell_size,
        max_steps=max_steps
    ) 