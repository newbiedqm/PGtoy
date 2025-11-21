"""
Simple example demonstrating REINFORCE on CartPole-v1
"""
from reinforce import REINFORCE
import matplotlib.pyplot as plt


def main():
    # Create REINFORCE agent for CartPole
    print("Creating REINFORCE agent for CartPole-v1...")
    agent = REINFORCE(env_name='CartPole-v1', hidden_dim=128, lr=1e-2, gamma=0.99)
    
    # Train for a small number of episodes as a quick example
    print("\nTraining for 500 episodes...")
    rewards = agent.train(num_episodes=500, print_every=50)
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Add moving average
    window = 50
    moving_avg = [sum(rewards[max(0, i-window):i+1]) / min(i+1, window) 
                  for i in range(len(rewards))]
    plt.plot(moving_avg, label=f'Moving Average (window={window})', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig('cartpole_example.png')
    print("\nTraining plot saved to cartpole_example.png")
    
    # Test the trained policy
    print("\nTesting trained policy...")
    agent.test(num_episodes=10)


if __name__ == '__main__':
    main()
