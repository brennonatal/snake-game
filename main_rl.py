"""Main entry point for Snake RL training.

This script demonstrates how to use the Snake RL implementation
for training Q-Learning and DQN agents.
"""

import argparse
import sys
import os
from typing import Optional

# Add the snake_rl package to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from snake_rl import (
    SnakeEnvironment, 
    TabularQAgent, 
    DQNAgent, 
    SnakeTrainer,
    QLearningConfig,
    DQNConfig,
    get_default_config,
    get_preset_config
)
from snake_rl.logging_setup import setup_logging, log_system_info
from snake_rl.utils import set_random_seeds


def train_tabular_q_learning(config: QLearningConfig, seed: int = 42) -> None:
    """Train a tabular Q-learning agent.
    
    Args:
        config: Q-learning configuration
        seed: Random seed for reproducibility
    """
    print("🐍 Starting Tabular Q-Learning Training")
    print("=" * 50)
    
    # Set random seeds
    set_random_seeds(seed)
    
    # Create environment
    env = SnakeEnvironment(
        width=config.grid_width * config.cell_size,
        height=config.grid_height * config.cell_size,
        cell_size=config.cell_size,
        headless=True
    )
    
    # Create agent
    agent = TabularQAgent(
        state_size=env.state_size,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon_start,
        epsilon_min=config.epsilon_end,
        epsilon_decay=config.epsilon_decay
    )
    
    # Create trainer
    trainer = SnakeTrainer(agent, env, config.to_dict())
    
    print(f"Environment: {config.grid_width}x{config.grid_height} grid")
    print(f"Agent: Tabular Q-Learning")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epsilon: {config.epsilon_start} → {config.epsilon_end}")
    print()
    
    # Train the agent
    results = trainer.train()
    
    # Print results
    print("\n🎉 Training Completed!")
    print("=" * 50)
    print(f"Episodes completed: {results['episodes_completed']}")
    print(f"Best score: {results['best_score']}")
    print(f"Average score: {results['average_score']:.2f}")
    print(f"Training time: {results['total_training_time']:.1f} seconds")
    
    # Plot results
    trainer.plot_results("training_results_tabular.png")
    
    return results


def train_dqn(config: DQNConfig, seed: int = 42) -> None:
    """Train a DQN agent.
    
    Args:
        config: DQN configuration
        seed: Random seed for reproducibility
    """
    print("🐍 Starting DQN Training")
    print("=" * 50)
    
    # Set random seeds
    set_random_seeds(seed)
    
    # Create environment
    env = SnakeEnvironment(
        width=config.grid_width * config.cell_size,
        height=config.grid_height * config.cell_size,
        cell_size=config.cell_size,
        headless=True
    )
    
    # Create agent
    agent = DQNAgent(
        state_size=env.state_size,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon_start,
        epsilon_min=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_frequency,
        device=config.device
    )
    
    # Create trainer
    trainer = SnakeTrainer(agent, env, config.to_dict())
    
    print(f"Environment: {config.grid_width}x{config.grid_height} grid")
    print(f"Agent: DQN")
    print(f"Network: {config.hidden_layers}")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Device: {agent.device}")
    print()
    
    # Train the agent
    results = trainer.train()
    
    # Print results
    print("\n🎉 Training Completed!")
    print("=" * 50)
    print(f"Episodes completed: {results['episodes_completed']}")
    print(f"Best score: {results['best_score']}")
    print(f"Average score: {results['average_score']:.2f}")
    print(f"Training time: {results['total_training_time']:.1f} seconds")
    
    # Plot results
    trainer.plot_results("training_results_dqn.png")
    
    return results


def test_environment() -> None:
    """Test the environment with random actions."""
    print("🧪 Testing Environment")
    print("=" * 30)
    
    env = SnakeEnvironment(headless=False)
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Random action
            action = env.action_space - 1  # Get a random action
            import random
            action = random.randint(0, env.action_space - 1)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode ended: {info['status']}")
                print(f"Score: {env.score}, Steps: {steps}, Total Reward: {total_reward:.1f}")
                break
            
            if steps > 200:  # Prevent infinite episodes
                break


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Snake RL Training")
    parser.add_argument("--algorithm", "-a", choices=["tabular", "dqn"], default="tabular",
                       help="RL algorithm to use")
    parser.add_argument("--config", "-c", choices=["default", "fast", "stable"], default="default",
                       help="Configuration preset")
    parser.add_argument("--episodes", "-e", type=int, help="Number of training episodes")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Test environment only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        log_system_info()
    
    # Test mode
    if args.test:
        test_environment()
        return
    
    # Get configuration
    if args.config == "default":
        config = get_default_config(args.algorithm)
    else:
        preset_name = f"{args.algorithm}_{args.config}"
        try:
            config = get_preset_config(preset_name)
        except ValueError:
            print(f"❌ Unknown preset: {preset_name}")
            print("Available presets: tabular_fast, tabular_stable, dqn_fast, dqn_stable")
            return
    
    # Override episodes if specified
    if args.episodes:
        config.max_episodes = args.episodes
    
    # Train the selected algorithm
    if args.algorithm == "tabular":
        train_tabular_q_learning(config, args.seed)
    elif args.algorithm == "dqn":
        train_dqn(config, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 