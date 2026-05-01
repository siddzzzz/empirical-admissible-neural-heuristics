import numpy as np

class TrajectoryBuffer:
    """
    A memory buffer to store sequences of states, actions, and rewards for the transformer planner.
    """
    def __init__(self, max_seq_len=20, obs_dim=144):
        self.max_seq_len = max_seq_len
        self.obs_dim = obs_dim
        
        # Current trajectory
        self.states = np.zeros((max_seq_len, obs_dim), dtype=np.float32)
        self.actions = np.zeros(max_seq_len, dtype=np.int32)
        self.rewards = np.zeros(max_seq_len, dtype=np.float32)
        self.ptr = 0

    def add(self, state, action, reward):
        if self.ptr < self.max_seq_len:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.ptr += 1
        else:
            # Shift left
            self.states[:-1] = self.states[1:]
            self.actions[:-1] = self.actions[1:]
            self.rewards[:-1] = self.rewards[1:]
            
            self.states[-1] = state
            self.actions[-1] = action
            self.rewards[-1] = reward

    def get_sequence(self):
        """Returns the padded sequence up to max_seq_len"""
        return self.states, self.actions, self.rewards, self.ptr

    def reset(self):
        self.states = np.zeros((self.max_seq_len, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.max_seq_len, dtype=np.int32)
        self.rewards = np.zeros(self.max_seq_len, dtype=np.float32)
        self.ptr = 0
