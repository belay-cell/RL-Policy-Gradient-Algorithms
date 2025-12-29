"""Utility functions and classes for RL algorithms."""

import random
from typing import List


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        """Add a transition to the buffer.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
