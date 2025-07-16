"""Visualization tools for Snake RL training and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizes training progress and metrics."""
    
    def __init__(self, save_dir: str = "plots"):
        """Initialize the training visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        self.colors = {
            'reward': '#2E86AB',
            'score': '#A23B72',
            'length': '#F18F01',
            'epsilon': '#C73E1D',
            'loss': '#8B2635',
            'q_value': '#5D737E'
        }
        
    def plot_training_metrics(self, 
                            metrics: Dict[str, List[float]], 
                            save_path: Optional[str] = None,
                            show_rolling_avg: bool = True,
                            rolling_window: int = 100) -> None:
        """Plot comprehensive training metrics.
        
        Args:
            metrics: Dictionary of metric lists
            save_path: Path to save the plot
            show_rolling_avg: Whether to show rolling averages
            rolling_window: Window size for rolling average
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
        
        # Episode rewards
        self._plot_metric(axes[0, 0], metrics.get('episode_rewards', []), 
                         'Episode Rewards', 'Episode', 'Total Reward',
                         self.colors['reward'], show_rolling_avg, rolling_window)
        
        # Episode scores
        self._plot_metric(axes[0, 1], metrics.get('episode_scores', []), 
                         'Episode Scores', 'Episode', 'Score (Food Eaten)',
                         self.colors['score'], show_rolling_avg, rolling_window)
        
        # Episode lengths
        self._plot_metric(axes[0, 2], metrics.get('episode_lengths', []), 
                         'Episode Lengths', 'Episode', 'Steps',
                         self.colors['length'], show_rolling_avg, rolling_window)
        
        # Exploration rate (epsilon)
        self._plot_metric(axes[1, 0], metrics.get('epsilon_history', []), 
                         'Exploration Rate', 'Episode', 'Epsilon',
                         self.colors['epsilon'], False)
        
        # Loss (if available)
        if 'loss_history' in metrics and metrics['loss_history']:
            self._plot_metric(axes[1, 1], metrics['loss_history'], 
                             'Training Loss', 'Update Step', 'Loss',
                             self.colors['loss'], show_rolling_avg, rolling_window // 10)
        else:
            axes[1, 1].text(0.5, 0.5, 'Loss data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Loss')
        
        # Q-values (if available)
        if 'q_value_history' in metrics and metrics['q_value_history']:
            self._plot_metric(axes[1, 2], metrics['q_value_history'], 
                             'Q-Values', 'Episode', 'Average Q-Value',
                             self.colors['q_value'], show_rolling_avg, rolling_window)
        else:
            axes[1, 2].text(0.5, 0.5, 'Q-value data not available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Q-Values')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved to {save_path}")
        
        plt.show()
    
    def _plot_metric(self, ax, data: List[float], title: str, xlabel: str, ylabel: str,
                    color: str, show_rolling_avg: bool = True, window: int = 100) -> None:
        """Plot a single metric with optional rolling average."""
        if not data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(title)
            return
        
        episodes = range(len(data))
        
        # Plot raw data
        ax.plot(episodes, data, alpha=0.6, color=color, linewidth=0.8, label='Raw')
        
        # Plot rolling average
        if show_rolling_avg and len(data) > window:
            rolling_avg = self._calculate_rolling_average(data, window)
            ax.plot(episodes, rolling_avg, color='red', linewidth=2, 
                   label=f'Rolling Avg ({window})')
            ax.legend()
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    def _calculate_rolling_average(self, data: List[float], window: int) -> List[float]:
        """Calculate rolling average of data."""
        rolling_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            rolling_avg.append(np.mean(data[start_idx:i+1]))
        return rolling_avg
    
    def plot_performance_comparison(self, 
                                  results: Dict[str, Dict[str, List[float]]], 
                                  metric: str = 'episode_scores',
                                  save_path: Optional[str] = None) -> None:
        """Compare performance across different algorithms or configurations.
        
        Args:
            results: Dictionary mapping algorithm names to metrics
            metric: Metric to compare
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for i, (name, metrics) in enumerate(results.items()):
            if metric in metrics:
                data = metrics[metric]
                episodes = range(len(data))
                color = list(self.colors.values())[i % len(self.colors)]
                
                # Plot rolling average
                if len(data) > 100:
                    rolling_avg = self._calculate_rolling_average(data, 100)
                    plt.plot(episodes, rolling_avg, label=name, color=color, linewidth=2)
                else:
                    plt.plot(episodes, data, label=name, color=color, linewidth=2)
        
        plt.title(f'Performance Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to {save_path}")
        
        plt.show()


class GameStateVisualizer:
    """Visualizes game states and agent behavior."""
    
    def __init__(self, grid_width: int, grid_height: int, cell_size: int = 10):
        """Initialize the game state visualizer.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            cell_size: Size of each cell in the visualization
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        
        # Colors for different game elements
        self.colors = {
            'empty': '#F0F0F0',
            'snake_head': '#2E7D32',
            'snake_body': '#4CAF50',
            'food': '#D32F2F',
            'wall': '#424242'
        }
    
    def visualize_state(self, state: np.ndarray, 
                       snake_positions: Optional[List[Tuple[int, int]]] = None,
                       food_position: Optional[Tuple[int, int]] = None,
                       title: str = "Game State") -> None:
        """Visualize a single game state.
        
        Args:
            state: State array (can be grid or feature-based)
            snake_positions: List of snake segment positions
            food_position: Food position
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(state) == self.grid_width * self.grid_height:
            # Grid-based state
            grid = state.reshape(self.grid_height, self.grid_width)
            self._draw_grid(ax, grid)
        else:
            # Feature-based state - need positions
            if snake_positions and food_position:
                grid = self._create_grid_from_positions(snake_positions, food_position)
                self._draw_grid(ax, grid)
            else:
                ax.text(0.5, 0.5, 'Cannot visualize feature-based state\nwithout positions', 
                       ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(title)
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_aspect('equal')
        plt.show()
    
    def _draw_grid(self, ax, grid: np.ndarray) -> None:
        """Draw the game grid."""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_value = grid[y, x]
                
                if cell_value == 0:  # Empty
                    color = self.colors['empty']
                elif cell_value == 1:  # Snake head
                    color = self.colors['snake_head']
                elif cell_value == 2:  # Snake body
                    color = self.colors['snake_body']
                elif cell_value == 3:  # Food
                    color = self.colors['food']
                else:
                    color = self.colors['wall']
                
                rect = Rectangle((x, self.grid_height - y - 1), 1, 1, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
    
    def _create_grid_from_positions(self, snake_positions: List[Tuple[int, int]], 
                                   food_position: Tuple[int, int]) -> np.ndarray:
        """Create grid from positions."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        # Mark snake body
        for i, (x, y) in enumerate(snake_positions):
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[y, x] = 1 if i == 0 else 2  # Head = 1, Body = 2
        
        # Mark food
        food_x, food_y = food_position
        if 0 <= food_x < self.grid_width and 0 <= food_y < self.grid_height:
            grid[food_y, food_x] = 3
        
        return grid
    
    def create_episode_animation(self, 
                               episode_states: List[np.ndarray],
                               episode_actions: List[int],
                               episode_rewards: List[float],
                               save_path: Optional[str] = None) -> None:
        """Create animation of an episode.
        
        Args:
            episode_states: List of states throughout the episode
            episode_actions: List of actions taken
            episode_rewards: List of rewards received
            save_path: Path to save animation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Initialize plots
        ax1.set_xlim(0, self.grid_width)
        ax1.set_ylim(0, self.grid_height)
        ax1.set_aspect('equal')
        ax1.set_title('Game State')
        
        ax2.set_title('Episode Progress')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cumulative Reward')
        
        # Animation function
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Draw current state
            if frame < len(episode_states):
                state = episode_states[frame]
                if len(state) == self.grid_width * self.grid_height:
                    grid = state.reshape(self.grid_height, self.grid_width)
                    self._draw_grid(ax1, grid)
            
            ax1.set_xlim(0, self.grid_width)
            ax1.set_ylim(0, self.grid_height)
            ax1.set_aspect('equal')
            ax1.set_title(f'Step {frame}')
            
            # Plot reward progress
            if frame > 0:
                cumulative_rewards = np.cumsum(episode_rewards[:frame])
                steps = range(len(cumulative_rewards))
                ax2.plot(steps, cumulative_rewards, 'b-')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Cumulative Reward')
                ax2.set_title(f'Total Reward: {cumulative_rewards[-1]:.1f}')
                ax2.grid(True)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(episode_states), 
                                     interval=200, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            logger.info(f"Episode animation saved to {save_path}")
        
        plt.show()


class PolicyVisualizer:
    """Visualizes learned policies and Q-values."""
    
    def __init__(self, grid_width: int, grid_height: int):
        """Initialize the policy visualizer.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.action_arrows = {
            0: '↑',  # UP
            1: '↓',  # DOWN
            2: '←',  # LEFT
            3: '→'   # RIGHT
        }
    
    def visualize_q_values(self, agent, sample_states: List[np.ndarray], 
                          save_path: Optional[str] = None) -> None:
        """Visualize Q-values for sample states.
        
        Args:
            agent: Trained agent
            sample_states: List of sample states to analyze
            save_path: Path to save the plot
        """
        n_states = len(sample_states)
        fig, axes = plt.subplots(2, min(n_states, 4), figsize=(16, 8))
        
        if n_states == 1:
            axes = [axes]
        elif n_states <= 4:
            axes = axes.reshape(2, -1)
        
        for i, state in enumerate(sample_states[:8]):  # Limit to 8 states
            row = i // 4
            col = i % 4
            
            if n_states <= 4:
                ax = axes[row] if n_states == 1 else axes[row, col]
            else:
                ax = axes[row, col]
            
            # Get Q-values for this state
            if hasattr(agent, 'q_table'):
                # Tabular agent
                state_key = agent._state_to_key(state)
                q_values = agent.q_table[state_key]
            else:
                # DQN agent
                import torch
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = agent.q_network(state_tensor).numpy()[0]
            
            # Create bar plot of Q-values
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            bars = ax.bar(actions, q_values, color=['red', 'blue', 'green', 'orange'])
            ax.set_title(f'Q-Values - State {i+1}')
            ax.set_ylabel('Q-Value')
            
            # Highlight best action
            best_action = np.argmax(q_values)
            bars[best_action].set_color('gold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Q-values visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_policy_heatmap(self, agent, sample_positions: List[Tuple[int, int]],
                               food_position: Tuple[int, int], 
                               save_path: Optional[str] = None) -> None:
        """Visualize policy as a heatmap showing preferred actions.
        
        Args:
            agent: Trained agent
            sample_positions: List of positions to sample
            food_position: Fixed food position for the analysis
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create grid for visualization
        policy_grid = np.zeros((self.grid_height, self.grid_width))
        
        for x, y in sample_positions:
            # Create a simple state with snake head at (x, y)
            snake_positions = [(x, y)]
            state = self._create_simple_state(snake_positions, food_position)
            
            # Get best action from agent
            action = agent.get_action(state)
            policy_grid[y, x] = action
        
        # Create heatmap
        im = ax.imshow(policy_grid, cmap='viridis', aspect='equal')
        
        # Add action arrows
        for x, y in sample_positions:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                action = int(policy_grid[y, x])
                arrow = self.action_arrows.get(action, '?')
                ax.text(x, y, arrow, ha='center', va='center', 
                       color='white', fontsize=12, fontweight='bold')
        
        # Mark food position
        food_x, food_y = food_position
        ax.plot(food_x, food_y, 'r*', markersize=15, label='Food')
        
        ax.set_title('Policy Heatmap (Preferred Actions)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Policy heatmap saved to {save_path}")
        
        plt.show()
    
    def _create_simple_state(self, snake_positions: List[Tuple[int, int]], 
                           food_position: Tuple[int, int]) -> np.ndarray:
        """Create a simple state representation."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        # Mark snake
        for i, (x, y) in enumerate(snake_positions):
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[y, x] = 1 if i == 0 else 2
        
        # Mark food
        food_x, food_y = food_position
        if 0 <= food_x < self.grid_width and 0 <= food_y < self.grid_height:
            grid[food_y, food_x] = 3
        
        return grid.flatten() 