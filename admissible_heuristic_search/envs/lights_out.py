import numpy as np
import random
from admissible_heuristic_search.common.env import CombinatorialPuzzle

class LightsOut(CombinatorialPuzzle):
    """
    Lights Out puzzle environment implementing the CombinatorialPuzzle interface.
    W, H represent the grid size (default 3x3 for fast training, but scalable).
    State representation: flat binary array of length W * H where 1 is On, 0 is Off.
    Goal: turn all lights Off (all zeros).
    """
    def __init__(self, W: int = 3, H: int = 3):
        self.W = W
        self.H = H
        self.L = W * H
        
        # Canonical solved state: all lights Off
        self._solved_state = np.zeros(self.L, dtype=np.int8)
        self.state = self._solved_state.copy()
        
        # Actions: clicking any of the W*H cells
        self._action_space_names = [f"CLICK_{i}" for i in range(self.L)]
        
    def reset(self) -> np.ndarray:
        self.state = self._solved_state.copy()
        return self.state
        
    def scramble(self, depth: int) -> np.ndarray:
        self.reset()
        # Click random cells (clicking a cell at most once since double clicks cancel out)
        clicked_cells = random.sample(range(self.L), min(depth, self.L))
        for cell in clicked_cells:
            self.step(cell)
        return self.state
        
    def step(self, action_idx: int) -> float:
        # Action is clicking cell `action_idx`
        r, c = action_idx // self.W, action_idx % self.W
        
        # Toggle clicked cell and its 4 orthogonal neighbors
        neighbors = [
            (r, c),
            (r-1, c),
            (r+1, c),
            (r, c-1),
            (r, c+1)
        ]
        
        for nr, nc in neighbors:
            if 0 <= nr < self.H and 0 <= nc < self.W:
                idx = nr * self.W + nc
                self.state[idx] ^= 1 # XOR with 1 to toggle
                
        return 1.0 # step cost is 1.0
        
    def get_state(self) -> np.ndarray:
        return self.state.copy()
        
    def set_state(self, state: np.ndarray):
        self.state = np.array(state, dtype=np.int8)
        
    def is_solved(self) -> bool:
        return np.all(self.state == 0)
        
    @property
    def action_space_names(self) -> list:
        return self._action_space_names
        
    @property
    def state_dim(self) -> int:
        return self.L
        
    @property
    def one_hot_dim(self) -> int:
        return self.L * 2
        
    def to_one_hot(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            one_hot = np.zeros((self.L, 2), dtype=np.float32)
            one_hot[np.arange(self.L), state] = 1.0
            return one_hot.flatten()
        else:
            B = state.shape[0]
            one_hot = np.zeros((B, self.L, 2), dtype=np.float32)
            one_hot[np.arange(B)[:, None], np.arange(self.L), state] = 1.0
            return one_hot.reshape(B, -1)
            
    def get_base_heuristic(self, state: np.ndarray) -> float:
        """
        Admissible heuristic: Since each click toggles at most 5 lights,
        the minimum number of clicks to resolve `k` active lights is ceil(k / 5).
        """
        active_lights = np.sum(state == 1)
        return float(np.ceil(active_lights / 5.0))
