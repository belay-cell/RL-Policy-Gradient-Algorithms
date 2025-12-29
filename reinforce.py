"""REINFORCE algorithm implementation with and without baseline."""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List


class REINFORCE:
    """Vanilla REINFORCE algorithm."""
    
    def __init__(self, policy_model, lr: float, solve_criteria: float, episode_limit: int):
        """
        Args:
            policy_model: Policy network
            lr: Learning rate
            solve_criteria: Average reward threshold to consider problem solved
            episode_limit: Maximum number of episodes to train
        """
        self.policy_model = policy_model
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.solve_criteria = solve_criteria
        self.episode_limit = episode_limit
        self.log_probs = []
        self.rewards = []
    
    def calculate_loss(self) -> torch.Tensor:
        """Calculate REINFORCE loss using Monte Carlo returns."""
        DISCOUNT_FACTOR = 0.99
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + DISCOUNT_FACTOR * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy gradient loss
        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss += -log_prob * Gt
        
        return loss
    
    def optimize(self):
        """Update policy parameters."""
        loss = self.calculate_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def interaction_step(self, state, env):
        """Interact with environment for one step."""
        action, log_prob = self.policy_model.full_pass(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = float(reward)
        
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        
        return new_state, done
    
    def train(self, env) -> List[float]:
        """Train the agent.
        
        Returns:
            List of total rewards per episode
        """
        total_rewards = []
        pbar = tqdm(range(1, self.episode_limit + 1))
        solved = False
        
        for episode in pbar:
            (state, _), done = env.reset(seed=54321), False
            self.log_probs = []
            self.rewards = []
            
            # Generate episode
            while not done:
                state, done = self.interaction_step(state, env)
            
            # Update policy
            self.optimize()
            
            # Track progress
            total_reward = sum(self.rewards)
            total_rewards.append(total_reward)
            avg_rewards = np.mean(total_rewards[-10:])
            pbar.set_description(
                f"reward avg: {avg_rewards:.2f} "
                f"min: {min(total_rewards):.2f} "
                f"max: {max(total_rewards):.2f}"
            )
            
            if avg_rewards >= self.solve_criteria:
                print(f"\nSOLVED in {episode} episodes!")
                solved = True
                break
        
        if not solved:
            raise Exception("Not solved. Please check your implementation!")
        
        return total_rewards


class REINFORCE_Baseline(REINFORCE):
    """REINFORCE with baseline to reduce variance."""
    
    def calculate_loss(self) -> torch.Tensor:
        """Calculate REINFORCE loss with baseline."""
        DISCOUNT_FACTOR = 0.99
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + DISCOUNT_FACTOR * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        
        # Use mean of returns as baseline
        baseline = returns.mean()
        advantages = returns - baseline
        
        # Calculate policy gradient loss with advantage
        loss = 0
        for log_prob, adv in zip(self.log_probs, advantages):
            loss += -log_prob * adv
        
        return loss
