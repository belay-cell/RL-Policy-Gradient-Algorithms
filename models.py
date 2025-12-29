"""Neural network models for policy and value function approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftmaxPolicy(nn.Module):
    """Softmax policy network for discrete action spaces."""
    
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: Size of state representation
            out_features: Size of action representation (number of actions)
        """
        super(SoftmaxPolicy, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def full_pass(self, state) -> Tuple[int, torch.Tensor]:
        """Get action and log probability for a given state."""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class Critic(nn.Module):
    """Critic network for value function approximation."""
    
    def __init__(self, in_features: int):
        """
        Args:
            in_features: Size of state representation
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPGActor(nn.Module):
    """Actor network for DDPG with continuous action spaces."""
    
    def __init__(self, in_features: int, out_features: int, action_range: float):
        """
        Args:
            in_features: Size of state representation
            out_features: Size of action representation
            action_range: Action range (e.g., action ∈ [-2, 2] if action_range is 2.0)
        """
        super(DDPGActor, self).__init__()
        self.action_range = action_range
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.squeeze(-1).unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x * self.action_range


class DDPGCritic(nn.Module):
    """Critic network for DDPG that takes both state and action as input."""
    
    def __init__(self, in_features_obs: int, in_features_act: int):
        """
        Args:
            in_features_obs: Size of state representation
            in_features_act: Size of action representation
        """
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(in_features_obs + in_features_act, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.squeeze(-1).unsqueeze(0)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
