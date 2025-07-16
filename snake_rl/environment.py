"""Snake Environment - RL Environment wrapper for the Snake game."""

import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
import logging
import sys
import os

# Add parent directory to path to import game_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine import SnakeGameEngine, GameStatus

logger = logging.getLogger(__name__)


class SnakeEnvironment:
    """Snake Game Environment for Reinforcement Learning.
    
    Wraps the existing Snake game into a standard RL environment interface
    following the OpenAI Gym style.
    """
    
    def __init__(self, 
                 width: int = 600, 
                 height: int = 400, 
                 cell_size: int = 10, 
                 headless: bool = True,
                 max_steps: Optional[int] = 1000,
                 reward_type: str = "shaped",
                 fast_mode: bool = False):
        """Initialize the Snake environment.
        
        Args:
            width: Game board width in pixels
            height: Game board height in pixels  
            cell_size: Size of each game cell in pixels
            headless: If True, disable rendering for faster training
            max_steps: Maximum steps per episode (None for unlimited)
            reward_type: Type of reward system ("sparse", "shaped", "strong_shaped", "potential")
            fast_mode: If True, optimize for speed with smaller board and simpler calculations
        """
        # Optimize for speed if requested
        if fast_mode:
            width = min(width, 300)  # Smaller board
            height = min(height, 200)  
            max_steps = min(max_steps or 1000, 200)  # Shorter episodes
            
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.headless = headless
        self.reward_type = reward_type
        self.fast_mode = fast_mode
        
        # Grid dimensions
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        
        # Action space: 4 discrete actions
        self.action_space = 4  # UP, DOWN, LEFT, RIGHT
        self.actions = {
            0: "UP",
            1: "DOWN", 
            2: "LEFT",
            3: "RIGHT"
        }
        
        # Initialize state processor
        from .state import StateProcessor
        self.state_processor = StateProcessor(self.grid_width, self.grid_height)
        
        # State space dimensions (24 features for enhanced representation)
        self.state_size = self.state_processor.state_size
        
        # Initialize game engine
        self.game_engine = SnakeGameEngine(
            width=width,
            height=height,
            cell_size=cell_size,
            max_steps=max_steps
        )
        
        # Initialize environment
        self.reset()
        
        logger.info(f"SnakeEnvironment initialized: {self.grid_width}x{self.grid_height} grid, headless={headless}, reward_type={reward_type}")
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset the game engine
        game_state = self.game_engine.reset()
        
        logger.debug("Environment reset")
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (0-3 for UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.game_engine.game_over:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        
        # Convert action to string
        action_str = self.actions[action]
        
        # Get current state for reward calculation
        old_state = self.get_state()
        old_distance = self._calculate_distance_to_food()
        
        # Execute action in game engine
        game_state, info = self.game_engine.step(action_str)
        
        # Calculate reward based on game status
        reward = self._calculate_reward(old_distance, info)
        
        # Check if episode is done
        done = game_state.game_over
        
        # Add additional info
        info.update({
            "steps": game_state.steps,
            "snake_length": len(game_state.snake_body),
            "food_position": (game_state.food_x, game_state.food_y),
            "head_position": (game_state.snake_head.x, game_state.snake_head.y)
        })
        
        return self.get_state(), reward, done, info
    
    def get_state(self) -> np.ndarray:
        """Get current state representation.
        
        Returns:
            Enhanced feature vector state representation (24 features)
        """
        game_state = self.game_engine.get_state()
        
        # Convert snake body to dictionary format for state processor
        snake_head = {"x": game_state.snake_head.x, "y": game_state.snake_head.y}
        snake_body = [{"x": segment.x, "y": segment.y} for segment in game_state.snake_body]
        food_pos = (game_state.food_x, game_state.food_y)
        direction = (game_state.x_move, game_state.y_move)
        
        # Use enhanced feature vector encoding
        return self.state_processor.encode_state(
            snake_head, snake_body, food_pos, direction, self.cell_size
        )
    
    def render(self, mode: str = "human") -> None:
        """Render the current state.
        
        Args:
            mode: Rendering mode ("human" for visualization, "rgb_array" for numpy array)
        """
        if not self.headless and mode == "human":
            # TODO: Implement pygame rendering for visualization
            # For now, print a simple ASCII representation
            self._render_ascii()
    
    def get_valid_actions(self) -> np.ndarray:
        """Get valid actions from current state.
        
        Returns:
            Array of valid action indices
        """
        valid_actions = self.game_engine.get_valid_actions()
        return np.array([list(self.actions.keys())[list(self.actions.values()).index(action)] 
                        for action in valid_actions])
    
    def _calculate_reward(self, old_distance: float, info: Dict[str, Any]) -> float:
        """Calculate reward based on game state and action outcome.
        
        Args:
            old_distance: Distance to food before action
            info: Game info dictionary
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        reward_components = {}
        
        # Primary rewards based on game status
        status = info.get("status", "normal")
        
        if status == GameStatus.FOOD_EATEN.value:
            reward_components['food_reward'] = 100.0
            reward += reward_components['food_reward']
            logger.debug("Food eaten reward: +100")
        elif status in [GameStatus.WALL_COLLISION.value, GameStatus.SELF_COLLISION.value]:
            reward_components['collision_penalty'] = -100.0
            reward += reward_components['collision_penalty']
            logger.debug(f"Collision penalty: -100 ({status})")
        elif status == GameStatus.MAX_STEPS.value:
            reward_components['timeout_penalty'] = -50.0
            reward += reward_components['timeout_penalty']
            logger.debug("Max steps penalty: -50")
        
        # Secondary rewards based on reward type
        if not self.game_engine.game_over:
            if self.reward_type == "sparse":
                # Only survival bonus, no distance shaping
                reward_components['survival_bonus'] = 0.1
                reward += reward_components['survival_bonus']
                
            elif self.reward_type == "shaped":
                # Current moderate distance shaping
                new_distance = self._calculate_distance_to_food()
                distance_change = old_distance - new_distance
                
                if distance_change > 0:
                    reward_components['distance_reward'] = 2.0
                    reward += reward_components['distance_reward']
                elif distance_change < 0:
                    reward_components['distance_penalty'] = -1.0
                    reward += reward_components['distance_penalty']
                else:
                    reward_components['no_progress'] = -0.1
                    reward += reward_components['no_progress']
                
                reward_components['survival_bonus'] = 0.1
                reward += reward_components['survival_bonus']
                
            elif self.reward_type == "strong_shaped":
                # Stronger distance signals for clearer learning
                new_distance = self._calculate_distance_to_food()
                distance_change = old_distance - new_distance
                
                if distance_change > 0:
                    reward_components['distance_reward'] = 5.0  # Stronger positive signal
                    reward += reward_components['distance_reward']
                elif distance_change < 0:
                    reward_components['distance_penalty'] = -2.0  # Stronger negative signal
                    reward += reward_components['distance_penalty']
                else:
                    reward_components['no_progress'] = -0.5  # Discourage wandering
                    reward += reward_components['no_progress']
                
                reward_components['survival_bonus'] = 0.1
                reward += reward_components['survival_bonus']
                
            elif self.reward_type == "potential":
                # Potential-based shaping (theoretically sound)
                new_distance = self._calculate_distance_to_food()
                max_distance = self.grid_width + self.grid_height
                
                # Potential function: closer to food = higher potential
                old_potential = 1.0 - (old_distance / max_distance)
                new_potential = 1.0 - (new_distance / max_distance)
                
                # Reward is difference in potential (scaled down)
                potential_reward = (new_potential - old_potential) * 10.0
                reward_components['potential_reward'] = potential_reward
                reward += potential_reward
                
                reward_components['survival_bonus'] = 0.1
                reward += reward_components['survival_bonus']
        
        # Log detailed reward breakdown for debugging
        if logger.isEnabledFor(logging.DEBUG):
            components_str = ", ".join([f"{k}={v:.1f}" for k, v in reward_components.items()])
            logger.debug(f"Reward components: {components_str}, Total: {reward:.1f}")
        
        return reward
    
    def _calculate_distance_to_food(self) -> float:
        """Calculate Manhattan distance from snake head to food.
        
        Returns:
            Distance to food
        """
        game_state = self.game_engine.get_state()
        head_x = game_state.snake_head.x
        head_y = game_state.snake_head.y
        food_x = game_state.food_x
        food_y = game_state.food_y
        
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _render_ascii(self) -> None:
        """Render ASCII representation of the game state."""
        game_state = self.game_engine.get_state()
        
        # Create ASCII grid
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Place snake body
        for segment in game_state.snake_body[1:]:
            grid_x = segment.x // self.cell_size
            grid_y = segment.y // self.cell_size
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                grid[grid_y][grid_x] = 'o'
        
        # Place snake head
        head_grid_x = game_state.snake_head.x // self.cell_size
        head_grid_y = game_state.snake_head.y // self.cell_size
        if 0 <= head_grid_x < self.grid_width and 0 <= head_grid_y < self.grid_height:
            grid[head_grid_y][head_grid_x] = 'H'
        
        # Place food
        food_grid_x = game_state.food_x // self.cell_size
        food_grid_y = game_state.food_y // self.cell_size
        if 0 <= food_grid_x < self.grid_width and 0 <= food_grid_y < self.grid_height:
            grid[food_grid_y][food_grid_x] = 'F'
        
        # Print grid
        print("\n" + "=" * (self.grid_width + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("=" * (self.grid_width + 2))
        print(f"Score: {game_state.score}, Steps: {game_state.steps}, Status: {game_state.status.value}")


class SnakeEnvironmentConfig:
    """Configuration class for SnakeEnvironment."""
    
    def __init__(self,
                 width: int = 600,
                 height: int = 400,
                 cell_size: int = 10,
                 headless: bool = True,
                 max_steps: Optional[int] = 1000):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.headless = headless
        self.max_steps = max_steps
    
    def create_environment(self) -> SnakeEnvironment:
        """Create environment with this configuration."""
        return SnakeEnvironment(
            width=self.width,
            height=self.height,
            cell_size=self.cell_size,
            headless=self.headless,
            max_steps=self.max_steps
        ) 