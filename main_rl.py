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


def train_tabular_q_learning(config: QLearningConfig, seed: int = 42, fast_mode: bool = False, model_name: str = None) -> None:
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
        headless=True,
        fast_mode=fast_mode
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
    
    # Set custom model name if provided
    if model_name:
        config_dict = config.to_dict()
        config_dict['model_name'] = model_name
    else:
        config_dict = config.to_dict()
    
    # Create trainer
    trainer = SnakeTrainer(agent, env, config_dict)
    
    print(f"Environment: {config.grid_width}x{config.grid_height} grid")
    print(f"Agent: Tabular Q-Learning")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epsilon: {config.epsilon_start} → {config.epsilon_end}")
    if model_name:
        print(f"Model save name: {model_name}")
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


def train_dqn(config: DQNConfig, seed: int = 42, fast_mode: bool = False, model_name: str = None) -> None:
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
        headless=True,
        fast_mode=fast_mode
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
    
    # Set custom model name if provided
    if model_name:
        config_dict = config.to_dict()
        config_dict['model_name'] = model_name
    else:
        config_dict = config.to_dict()
    
    # Create trainer
    trainer = SnakeTrainer(agent, env, config_dict)
    
    print(f"Environment: {config.grid_width}x{config.grid_height} grid")
    print(f"Agent: DQN")
    print(f"Network: {config.hidden_layers}")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Device: {agent.device}")
    if model_name:
        print(f"Model save name: {model_name}")
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
    
    import random
    import time
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Random action
            action = random.randint(0, env.action_space - 1)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Add small delay to make it visible
            time.sleep(0.1)
            
            if done:
                print(f"Episode ended: {info['status']}")
                # Get score from game engine state
                game_state = env.game_engine.get_state()
                print(f"Score: {game_state.score}, Steps: {steps}, Total Reward: {total_reward:.1f}")
                break
            
            if steps > 200:  # Prevent infinite episodes
                print(f"Episode timeout after {steps} steps")
                break


def test_trained_model(model_path: str = None) -> None:
    """Test a trained DQN model.
    
    Args:
        model_path: Optional path to specific model file
    """
    print("🤖 Testing Trained DQN Model")
    print("=" * 40)
    
    # Load the lightning config to match training
    config = get_preset_config("dqn_lightning")
    
    # Create environment (headless for speed, we'll just see results)
    env = SnakeEnvironment(
        width=config.grid_width * config.cell_size,
        height=config.grid_height * config.cell_size,
        cell_size=config.cell_size,
        headless=True  # Fast testing
    )
    
    # Create DQN agent with same config as training (including network architecture)
    from snake_rl.agent import QNetwork
    import torch
    
    # Create agent with exact same architecture as training
    agent = DQNAgent(
        state_size=env.state_size,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=0.0,  # No exploration for testing
        epsilon_min=0.0,
        epsilon_decay=1.0,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_frequency,
        device=config.device
    )
    
    # Replace networks with correct architecture
    agent.q_network = QNetwork(env.state_size, 4, config.hidden_layers).to(agent.device)
    agent.target_network = QNetwork(env.state_size, 4, config.hidden_layers).to(agent.device)
    
    # Try to load a trained model
    import os
    import torch
    
    # Use provided model path or try to find one
    if model_path:
        checkpoint_path = model_path
    else:
        # Try to find the most recent model
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            models = [f for f in os.listdir(checkpoint_dir) if f.endswith('_final.pth')]
            if models:
                # Use the most recent final model
                checkpoint_path = os.path.join(checkpoint_dir, sorted(models)[-1])
                print(f"Auto-detected model: {checkpoint_path}")
            else:
                checkpoint_path = "checkpoints/agent_final.pth"  # fallback
        else:
            checkpoint_path = "checkpoints/agent_final.pth"  # fallback
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained model from {checkpoint_path}")
        try:
            # First peek at the model to detect architecture
            checkpoint = torch.load(checkpoint_path, map_location=agent.device)
            first_layer_shape = checkpoint['q_network_state_dict']['network.0.weight'].shape
            
            if first_layer_shape[0] == 512:
                # This is a [512, 256, 128] model
                print("Detected [512, 256, 128] architecture")
                agent.q_network = QNetwork(env.state_size, 4, [512, 256, 128]).to(agent.device)
                agent.target_network = QNetwork(env.state_size, 4, [512, 256, 128]).to(agent.device)
            elif first_layer_shape[0] == 128:
                # This is a [128, 64] model (lightning)
                print("Detected [128, 64] architecture (Lightning)")
                # Already correct architecture
            
            agent.load(checkpoint_path)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using random weights instead.")
    else:
        print("⚠️  No trained model found. Testing with random weights.")
        print("   Run training first: python main_rl.py --algorithm dqn --config lightning")
    
    print(f"Environment: {config.grid_width}x{config.grid_height} grid")
    print(f"Agent: DQN (no exploration)")
    print(f"Device: {agent.device}")
    print()
    
    # Test for multiple episodes
    test_episodes = 10
    scores = []
    rewards = []
    episode_lengths = []
    
    print(f"Running {test_episodes} test episodes...")
    
    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Use trained agent (no exploration)
            action = agent.get_action(state)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                game_state = env.game_engine.get_state()
                score = game_state.score
                scores.append(score)
                rewards.append(total_reward)
                episode_lengths.append(steps)
                
                print(f"Episode {episode + 1:2d}: Score={score:2d}, Steps={steps:3d}, Reward={total_reward:6.1f}, Status={info['status']}")
                break
            
            if steps > 500:  # Longer timeout for trained model
                print(f"Episode {episode + 1:2d}: TIMEOUT after {steps} steps")
                break
    
    # Calculate statistics
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        avg_reward = sum(rewards) / len(rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        
        print("\n📊 Test Results:")
        print("=" * 40)
        print(f"Average Score: {avg_score:.2f}")
        print(f"Max Score: {max_score}")
        print(f"Average Reward: {avg_reward:.1f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Episodes Completed: {len(scores)}")
        
        # Performance analysis
        good_games = len([s for s in scores if s >= 5])
        print(f"Games with Score ≥ 5: {good_games}/{len(scores)} ({100*good_games/len(scores):.1f}%)")


def get_available_models():
    """Get list of available trained models."""
    import os
    checkpoint_dir = "checkpoints"
    models = []
    
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        for file in files:
            if file.endswith('.pth'):
                # Get file info
                filepath = os.path.join(checkpoint_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                mtime = os.path.getmtime(filepath)
                models.append({
                    'name': file,
                    'path': filepath,
                    'size_mb': size_mb,
                    'mtime': mtime
                })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['mtime'], reverse=True)
    return models


def get_interactive_config():
    """Get configuration interactively from user."""
    print("🐍 Snake RL Interactive Mode")
    print("=" * 40)
    
    # Mode selection
    print("\n1. Select Mode:")
    print("   1) Train a new model")
    print("   2) Test an existing model [DEFAULT]")
    mode_choice = input("Enter choice (1-2) [2]: ").strip()
    mode = "train" if mode_choice == "1" else "test"
    
    if mode == "test":
        # Show available models
        models = get_available_models()
        if not models:
            print("\n❌ No trained models found!")
            print("   Please train a model first with: python main_rl.py")
            return None, None, None, None, None, True  # test_mode=True
        
        print(f"\n2. Select Model to Test:")
        for i, model in enumerate(models):
            # Extract info from filename
            name = model['name']
            size = model['size_mb']
            import time
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(model['mtime']))
            print(f"   {i+1}) {name} ({size:.1f}MB, {date})")
        
        model_choice = input(f"Enter choice (1-{len(models)}) [1]: ").strip()
        model_idx = int(model_choice) - 1 if model_choice else 0
        model_idx = max(0, min(model_idx, len(models) - 1))
        
        selected_model = models[model_idx]['path']
        print(f"\n✅ Selected: {models[model_idx]['name']}")
        
        return None, None, None, None, selected_model, True  # test_mode=True
    
    else:
        # Training configuration
        print("\n2. Select Algorithm:")
        print("   1) Tabular Q-Learning")
        print("   2) DQN (Deep Q-Network) [DEFAULT]")
        algo_choice = input("Enter choice (1-2) [2]: ").strip()
        algorithm = "dqn" if algo_choice in ["", "2"] else "tabular"
        
        # Config preset selection
        print(f"\n3. Select {algorithm.upper()} Configuration:")
        if algorithm == "dqn":
            print("   1) Lightning (fast, 100 episodes) [DEFAULT]")
            print("   2) Fast (5k episodes)")
            print("   3) Stable (15k episodes)")
            config_choice = input("Enter choice (1-3) [1]: ").strip()
            config_map = {"": "lightning", "1": "lightning", "2": "fast", "3": "stable"}
            config = config_map.get(config_choice, "lightning")
        else:
            print("   1) Learning (2k episodes) [DEFAULT]")
            print("   2) Fast (5k episodes)")
            print("   3) Stable (15k episodes)")
            config_choice = input("Enter choice (1-3) [1]: ").strip()
            config_map = {"": "learning", "1": "learning", "2": "fast", "3": "stable"}
            config = config_map.get(config_choice, "learning")
        
        # Episodes override
        preset_config = get_preset_config(f"{algorithm}_{config}")
        default_episodes = 100 if config == "lightning" else preset_config.max_episodes
        episodes_input = input(f"\n4. Number of episodes [{default_episodes}]: ").strip()
        episodes = int(episodes_input) if episodes_input else default_episodes
        
        # Seed
        seed_input = input("5. Random seed [42]: ").strip()
        seed = int(seed_input) if seed_input else 42
        
        # Model save name
        default_name = f"{algorithm}_{config}_{episodes}ep"
        model_name = input(f"6. Model save name [{default_name}]: ").strip()
        model_name = model_name if model_name else default_name
        
        return algorithm, config, episodes, seed, model_name, False  # test_mode=False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Snake RL Training")
    parser.add_argument("--algorithm", "-a", choices=["tabular", "dqn"], default="dqn",
                       help="RL algorithm to use (default: dqn)")
    parser.add_argument("--config", "-c", choices=["default", "fast", "stable", "learning", "lightning"], default="lightning",
                       help="Configuration preset (default: lightning)")
    parser.add_argument("--episodes", "-e", type=int, default=100, help="Number of training episodes (default: 100)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model-name", "-m", type=str, help="Custom model save name")
    parser.add_argument("--test", action="store_true", help="Test environment only")
    parser.add_argument("--test-model", action="store_true", help="Test trained DQN model")
    parser.add_argument("--model-path", type=str, help="Path to model for testing")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Interactive mode (triggered by --interactive flag or no arguments)
    if args.interactive or len(sys.argv) == 1:
        result = get_interactive_config()
        if result is None:
            return
        
        algorithm, config, episodes, seed, model_name, test_mode = result
        
        if test_mode:
            # Test mode selected
            test_trained_model(model_name)  # model_name contains the path in test mode
            return
        else:
            # Training mode selected
            args.algorithm = algorithm
            args.config = config
            args.episodes = episodes
            args.seed = seed
            args.model_name = model_name
    
    # Setup logging
    setup_logging()
    if args.verbose:
        log_system_info()
    
    # Test modes
    if args.test:
        test_environment()
        return
    
    if args.test_model:
        test_trained_model(args.model_path)
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
            print("Available presets: tabular_fast, tabular_stable, tabular_learning, dqn_fast, dqn_stable, dqn_lightning")
            return
    
    # Override episodes if specified
    if args.episodes:
        config.max_episodes = args.episodes
    
    # Generate default model name if not provided
    if not args.model_name:
        args.model_name = f"{args.algorithm}_{args.config}_{config.max_episodes}ep"
    
    # Train the selected algorithm  
    fast_mode = args.config == "lightning"
    if args.algorithm == "tabular":
        train_tabular_q_learning(config, args.seed, fast_mode, args.model_name)
    elif args.algorithm == "dqn":
        train_dqn(config, args.seed, fast_mode, args.model_name)


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