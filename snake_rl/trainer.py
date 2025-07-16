"""Training infrastructure for Snake RL."""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Any, Optional, Union
import logging
from collections import deque

from .environment import SnakeEnvironment
from .agent import TabularQAgent, DQNAgent

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Tracks and manages training metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize training metrics.
        
        Args:
            window_size: Size of rolling window for averages
        """
        self.window_size = window_size
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.episode_times = []
        
        # Running averages
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        self.score_window = deque(maxlen=window_size)
        
        # Training metrics
        self.epsilon_history = []
        self.loss_history = []
        self.q_value_history = []
        
        # Best performance tracking
        self.best_score = 0
        self.best_reward = float('-inf')
        
        logger.info(f"TrainingMetrics initialized with window size {window_size}")
    
    def record_episode(self, 
                      episode: int,
                      reward: float, 
                      length: int, 
                      score: int,
                      epsilon: float,
                      episode_time: float,
                      info: Dict[str, Any]) -> None:
        """Record metrics for a completed episode.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
            score: Game score (food eaten)
            epsilon: Current exploration rate
            episode_time: Time taken for episode
            info: Additional episode information
        """
        # Store episode metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)
        self.episode_times.append(episode_time)
        self.epsilon_history.append(epsilon)
        
        # Update rolling windows
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.score_window.append(score)
        
        # Update best performance
        if score > self.best_score:
            self.best_score = score
            logger.info(f"New best score: {score} in episode {episode}")
        
        if reward > self.best_reward:
            self.best_reward = reward
            logger.info(f"New best reward: {reward:.2f} in episode {episode}")
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get current average metrics.
        
        Returns:
            Dictionary of average metrics
        """
        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_length': np.mean(self.length_window) if self.length_window else 0,
            'avg_score': np.mean(self.score_window) if self.score_window else 0,
            'best_score': self.best_score,
            'best_reward': self.best_reward
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training progress curves.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7, label='Episode Reward')
        if len(self.reward_window) > 0:
            # Plot rolling average
            avg_rewards = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - self.window_size + 1)
                avg_rewards.append(np.mean(self.episode_rewards[start_idx:i+1]))
            axes[0, 0].plot(avg_rewards, color='red', label=f'Rolling Avg ({self.window_size})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode scores
        axes[0, 1].plot(self.episode_scores, alpha=0.7, label='Episode Score')
        if len(self.score_window) > 0:
            avg_scores = []
            for i in range(len(self.episode_scores)):
                start_idx = max(0, i - self.window_size + 1)
                avg_scores.append(np.mean(self.episode_scores[start_idx:i+1]))
            axes[0, 1].plot(avg_scores, color='red', label=f'Rolling Avg ({self.window_size})')
        axes[0, 1].set_title('Episode Scores (Food Eaten)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths, alpha=0.7, label='Episode Length')
        if len(self.length_window) > 0:
            avg_lengths = []
            for i in range(len(self.episode_lengths)):
                start_idx = max(0, i - self.window_size + 1)
                avg_lengths.append(np.mean(self.episode_lengths[start_idx:i+1]))
            axes[1, 0].plot(avg_lengths, color='red', label=f'Rolling Avg ({self.window_size})')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Exploration rate
        axes[1, 1].plot(self.epsilon_history, color='green')
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()


class SnakeTrainer:
    """Main training class for Snake RL agents."""
    
    def __init__(self,
                 agent: Union[TabularQAgent, DQNAgent],
                 environment: SnakeEnvironment,
                 config: Dict[str, Any]):
        """Initialize the trainer.
        
        Args:
            agent: RL agent to train
            environment: Snake game environment
            config: Training configuration
        """
        self.agent = agent
        self.env = environment
        self.config = config
        
        # Training metrics
        self.metrics = TrainingMetrics(config.get('metrics_window', 100))
        
        # Training parameters
        self.max_episodes = config.get('max_episodes', 10000)
        self.eval_frequency = config.get('eval_frequency', 100)
        self.save_frequency = config.get('save_frequency', 500)
        self.log_frequency = config.get('log_frequency', 50)
        
        # Paths
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.log_dir = config.get('log_dir', 'logs')
        
        # Training state
        self.current_episode = 0
        self.training_start_time = None
        
        logger.info(f"SnakeTrainer initialized for {self.max_episodes} episodes")
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            Training results summary
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.training_start_time = time.time()
        logger.info("Starting training...")
        
        try:
            for episode in range(self.current_episode, self.max_episodes):
                self.current_episode = episode
                episode_start_time = time.time()
                
                # Run episode
                episode_reward, episode_length, episode_score, episode_info = self._run_episode()
                episode_time = time.time() - episode_start_time
                
                # Record metrics
                self.metrics.record_episode(
                    episode=episode,
                    reward=episode_reward,
                    length=episode_length,
                    score=episode_score,
                    epsilon=self.agent.epsilon,
                    episode_time=episode_time,
                    info=episode_info
                )
                
                # Logging
                if episode % self.log_frequency == 0:
                    self._log_progress(episode)
                
                # Evaluation
                if episode % self.eval_frequency == 0 and episode > 0:
                    self._evaluate_agent(episode)
                
                # Save checkpoint
                if episode % self.save_frequency == 0 and episode > 0:
                    self._save_checkpoint(episode)
                
                # Check for early stopping
                if self._should_stop_early():
                    logger.info(f"Early stopping triggered at episode {episode}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final evaluation and save
            self._evaluate_agent(self.current_episode, final_eval=True)
            self._save_checkpoint(self.current_episode, final=True)
            
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self._get_training_summary()
    
    def _run_episode(self) -> tuple:
        """Run a single training episode.
        
        Returns:
            Tuple of (total_reward, episode_length, score, info)
        """
        state = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        while True:
            # Agent selects action
            action = self.agent.get_action(state)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Agent learns
            self.agent.update(state, action, reward, next_state, done)
            
            # Update tracking
            state = next_state
            total_reward += reward
            episode_length += 1
            
            if done:
                break
        
        return total_reward, episode_length, self.env.game_engine.score, info
    
    def _evaluate_agent(self, episode: int, num_eval_episodes: int = 10, final_eval: bool = False) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            episode: Current training episode
            num_eval_episodes: Number of evaluation episodes
            final_eval: Whether this is the final evaluation
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating agent at episode {episode}...")
        
        # Temporarily disable exploration
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        eval_rewards = []
        eval_scores = []
        eval_lengths = []
        
        for _ in range(num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_scores.append(self.env.game_engine.score)
            eval_lengths.append(episode_length)
        
        # Restore exploration
        self.agent.epsilon = original_epsilon
        
        # Calculate evaluation metrics
        eval_results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_score': np.mean(eval_scores),
            'std_score': np.std(eval_scores),
            'avg_length': np.mean(eval_lengths),
            'max_score': np.max(eval_scores),
            'min_score': np.min(eval_scores)
        }
        
        logger.info(f"Evaluation results: avg_score={eval_results['avg_score']:.2f}±{eval_results['std_score']:.2f}, "
                   f"max_score={eval_results['max_score']}, avg_reward={eval_results['avg_reward']:.2f}")
        
        return eval_results
    
    def _log_progress(self, episode: int) -> None:
        """Log training progress.
        
        Args:
            episode: Current episode number
        """
        avg_metrics = self.metrics.get_average_metrics()
        
        elapsed_time = time.time() - self.training_start_time
        episodes_per_second = episode / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Episode {episode:5d} | "
            f"Avg Score: {avg_metrics['avg_score']:5.2f} | "
            f"Avg Reward: {avg_metrics['avg_reward']:7.2f} | "
            f"Best Score: {avg_metrics['best_score']:3d} | "
            f"Epsilon: {self.agent.epsilon:.3f} | "
            f"Speed: {episodes_per_second:.1f} ep/s"
        )
    
    def _save_checkpoint(self, episode: int, final: bool = False) -> None:
        """Save training checkpoint.
        
        Args:
            episode: Current episode number
            final: Whether this is the final checkpoint
        """
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'agent_{suffix}.pth')
        
        self.agent.save(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.agent.load(checkpoint_path)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early.
        
        Returns:
            True if training should stop
        """
        # Implement early stopping criteria
        avg_metrics = self.metrics.get_average_metrics()
        
        # Stop if performance target is reached
        if avg_metrics['avg_score'] >= self.config.get('target_score', float('inf')):
            return True
        
        # Stop if no improvement for a long time
        # (Could implement more sophisticated early stopping here)
        
        return False
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results.
        
        Returns:
            Training summary dictionary
        """
        avg_metrics = self.metrics.get_average_metrics()
        total_time = time.time() - self.training_start_time
        
        return {
            'episodes_completed': self.current_episode,
            'total_training_time': total_time,
            'average_score': avg_metrics['avg_score'],
            'best_score': avg_metrics['best_score'],
            'average_reward': avg_metrics['avg_reward'],
            'best_reward': avg_metrics['best_reward'],
            'final_epsilon': self.agent.epsilon
        }
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot training results.
        
        Args:
            save_path: Path to save plots (optional)
        """
        self.metrics.plot_training_curves(save_path) 