import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete action spaces."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs


class REINFORCE:
    """REINFORCE (Monte Carlo Policy Gradient) algorithm."""
    
    def __init__(self, env_name, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.env = gym.make(env_name)
        self.gamma = gamma
        
        # Get state and action dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Initialize policy network and optimizer
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for episode data
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action using the policy network."""
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy gradient loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return policy_loss.item()
    
    def train(self, num_episodes=1000, print_every=100):
        """Train the policy."""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            # Collect episode trajectory
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rewards.append(reward)
                episode_reward += reward
            
            # Update policy at end of episode
            loss = self.update_policy()
            episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(episode_rewards[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Loss: {loss:.4f}")
        
        return episode_rewards
    
    def test(self, num_episodes=10, render=False):
        """Test the trained policy."""
        test_rewards = []
        
        for episode in range(num_episodes):
            if render:
                test_env = gym.make(self.env.spec.id, render_mode="human")
            else:
                test_env = self.env
                
            state, _ = test_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    probs = self.policy(state_tensor)
                action = torch.argmax(probs, dim=-1).item()
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            test_rewards.append(episode_reward)
            
            if render:
                test_env.close()
        
        avg_reward = np.mean(test_rewards)
        print(f"\nTest Results over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f} ± {np.std(test_rewards):.2f}")
        
        return test_rewards
