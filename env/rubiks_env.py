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
        
        # 18 primitive moves + any registered macros
        num_actions = len(self.cube.action_space_names)
        self.action_space = spaces.Discrete(num_actions)
        
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
        self.num_actions_taken = 0
        self.achieved_subgoals = {
            'first_layer': False,
            'oll': False
        }
        
        if scramble_length > 0:
            self.cube.scramble(scramble_length)
            
        # Re-evaluate initial state (in case scramble length was 0)
        self.achieved_subgoals['first_layer'] = self.cube.is_first_layer_solved()
        self.achieved_subgoals['oll'] = self.cube.is_oll_solved()
            
        obs = self._get_obs()
        info = {'is_solved': self.cube.is_solved()}
        return obs, info

    def step(self, action):
        old_state = self.cube.get_state()
        new_state = self.cube.step(action)
        
        action_name = self.cube.action_space_names[action]
        move_cost = self.cube.action_costs.get(action_name, 1)
        
        # Check subgoals
        is_solved = self.cube.is_solved()
        first_layer = self.cube.is_first_layer_solved()
        oll = self.cube.is_oll_solved()
        
        reward = compute_reward(
            old_state, new_state, is_solved,
            move_cost=move_cost,
            achieved_subgoals=self.achieved_subgoals,
            current_first_layer=first_layer,
            current_oll=oll
        )
        
        # Update trackers
        if first_layer: self.achieved_subgoals['first_layer'] = True
        if oll: self.achieved_subgoals['oll'] = True
        
        # Increment RL action counter
        self.num_actions_taken += 1
        
        # For Rubik's cube, episode is done when solved
        terminated = is_solved
        
        # Truncate if we exceed the maximum allowed RL decisions
        # We give the agent a minimum of 20 decisions, or double the scramble depth for harder scrambles.
        max_allowed_actions = max(20, self.scramble_moves * 2)
        truncated = self.num_actions_taken >= max_allowed_actions
        
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
