import argparse
import matplotlib.pyplot as plt
from reinforce import REINFORCE


def plot_rewards(rewards, save_path='training_rewards.png'):
    """Plot training rewards over episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards over Episodes')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train REINFORCE on Gymnasium environments')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gymnasium environment name (default: CartPole-v1)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension (default: 128)')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes (default: 10)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    
    args = parser.parse_args()
    
    print(f"Training REINFORCE on {args.env}")
    print(f"Hyperparameters:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Hidden Dimension: {args.hidden_dim}")
    print()
    
    # Create and train agent
    agent = REINFORCE(
        env_name=args.env,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma
    )
    
    # Train
    print("Starting training...")
    rewards = agent.train(num_episodes=args.episodes, print_every=100)
    
    # Plot results
    plot_rewards(rewards)
    
    # Test
    print("\nTesting trained policy...")
    agent.test(num_episodes=args.test_episodes, render=args.render)


if __name__ == '__main__':
    main()
