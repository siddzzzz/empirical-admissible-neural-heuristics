import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.cube import Cube2x2
from env.rewards import compute_reward

class RubiksEnv(gym.Env):
    """
    Gymnasium environment for a 2x2 Rubik's Cube.
    Observation is a one-hot encoded state of the cube (24 stickers x 6 colors = 144 values).
    Action space is Discrete(18).
    """
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, scramble_moves=5, render_mode=None):
        super(RubiksEnv, self).__init__()
        self.cube = Cube2x2()
        self.scramble_moves = scramble_moves
        self.render_mode = render_mode
        
        # 18 moves: U, U', U2, D, D', D2, etc.
        self.action_space = spaces.Discrete(18)
        
        # One-hot encoding of 24 stickers * 6 colors
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(144,), dtype=np.float32)

    def set_scramble_moves(self, moves):
        self.scramble_moves = moves

    def _get_obs(self):
        state = self.cube.get_state()
        # Create one-hot encoding
        one_hot = np.zeros((24, 6), dtype=np.float32)
        one_hot[np.arange(24), state] = 1.0
        return one_hot.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Use options to change curriculum scramble difficulty dynamically
        scramble_length = self.scramble_moves
        if options and 'scramble_moves' in options:
            scramble_length = options['scramble_moves']
            
        self.cube.reset()
        self.current_step = 0
        
        if scramble_length > 0:
            self.cube.scramble(scramble_length)
            
        obs = self._get_obs()
        info = {'is_solved': self.cube.is_solved()}
        return obs, info

    def step(self, action):
        old_state = self.cube.get_state()
        new_state = self.cube.step(action)
        
        is_solved = self.cube.is_solved()
        reward = compute_reward(old_state, new_state, is_solved)
        
        # Increment step counter
        self.current_step += 1
        
        # For Rubik's cube, episode is done when solved
        terminated = is_solved
        
        # Truncate if we exceed the maximum allowed steps (e.g. scramble moves + 10 buffer)
        max_allowed_steps = max(10, self.scramble_moves + 10)
        truncated = self.current_step >= max_allowed_steps
        
        obs = self._get_obs()
        info = {
            'is_solved': is_solved,
            'cube_state': new_state.copy()
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            return str(self.cube.state)
        elif self.render_mode == 'human':
            print(f"State: {self.cube.state}")
