"""Advantage Actor-Critic (A2C) algorithm implementation."""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


class A2C:
    """A2C algorithm with online learning."""
    
    def __init__(self, actor_model, critic_model, actor_lr: float, critic_lr: float, 
                 solve_criteria: float, episode_limit: int):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_lr)
        self.solve_criteria = solve_criteria
        self.episode_limit = episode_limit
        self.rewards = []
    
    def calculate_loss(self, state, log_prob, reward, next_state, done) -> Tuple[torch.Tensor, torch.Tensor]:
        DISCOUNT_FACTOR = 0.99
        
        state_value = self.critic_model(state)
        next_state_value = self.critic_model(next_state).detach()
        target_value = reward + (1 - done) * DISCOUNT_FACTOR * next_state_value
        advantage = target_value - state_value
        
        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        
        return actor_loss.mean(), critic_loss.mean()
    
    def optimize(self, state, log_prob, reward, new_state, done):
        actor_loss, critic_loss = self.calculate_loss(state, log_prob, reward, new_state, done)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def interaction_step(self, state, env):
        action, log_prob = self.actor_model.full_pass(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = float(reward)
        self.rewards.append(reward)
        return log_prob, reward, new_state, done
    
    def train(self, env) -> List[float]:
        total_rewards = []
        pbar = tqdm(range(1, self.episode_limit + 1))
        solved = False
        
        for episode in pbar:
            (state, _), done = env.reset(seed=54321), False
            self.rewards = []
            
            while not done:
                log_prob, reward, new_state, done = self.interaction_step(state, env)
                self.optimize(state, log_prob, reward, new_state, done)
                state = new_state
            
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
