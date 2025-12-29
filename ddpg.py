"""Deep Deterministic Policy Gradient (DDPG) algorithm."""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List
from copy import deepcopy


class DDPG:
    """DDPG for continuous action spaces."""
    
    def __init__(self, buffer, actor, critic, actor_lr: float, critic_lr: float, 
                 solve_criteria: float, episode_limit: int):
        self.buffer = buffer
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.solve_criteria = solve_criteria
        self.episode_limit = episode_limit
        self.rewards = []
    
    @staticmethod
    def soft_target_update(net, target_net, tau: float):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def update(self):
        BATCH_SIZE = 32
        DISCOUNT_FACTOR = 0.99
        TAU = 0.005
        
        if len(self.buffer) < BATCH_SIZE:
            return
        
        transitions = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            target_q = rewards + DISCOUNT_FACTOR * (1 - dones) * target_q_values
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_target_update(self.actor, self.target_actor, TAU)
        self.soft_target_update(self.critic, self.target_critic, TAU)
    
    def interaction_step(self, state, env):
        action = self.actor(state)
        action = action.detach().numpy().flatten()
        
        noise = np.random.normal(0, 0.1, size=action.shape)
        action = action + noise
        action = np.clip(action, -self.actor.action_range, self.actor.action_range)
        
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = float(reward)
        
        return action, reward, new_state, done
    
    def train(self, env) -> List[float]:
        total_rewards = []
        pbar = tqdm(range(1, self.episode_limit + 1))
        solved = False
        
        for episode in pbar:
            (state, _), done = env.reset(seed=54321), False
            self.rewards = []
            
            while not done:
                action, reward, new_state, done = self.interaction_step(state, env)
                self.buffer.push((state, action, reward, new_state, done))
                self.update()
                state = new_state
                self.rewards.append(reward)
            
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
