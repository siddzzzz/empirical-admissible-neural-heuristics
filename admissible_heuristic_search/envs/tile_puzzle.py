import numpy as np
import random
from admissible_heuristic_search.common.env import CombinatorialPuzzle

class TilePuzzle(CombinatorialPuzzle):
    """
    Sliding Tile Puzzle (8-puzzle or 15-puzzle) implementing the CombinatorialPuzzle interface.
    N represents grid size: N=3 for 8-puzzle, N=4 for 15-puzzle.
    State representation: flat array of length N^2 with values in [0, N^2-1] where 0 is the blank space.
    """
    def __init__(self, N: int = 3):
        self.N = N
        self.L = N * N
        
        # Canonical solved state: [1, 2, ..., L-1, 0]
        self._solved_state = np.zeros(self.L, dtype=np.int8)
        self._solved_state[:-1] = np.arange(1, self.L)
        self._solved_state[-1] = 0
        
        self.state = self._solved_state.copy()
        
        # Actions: Up, Down, Left, Right (moving the blank space)
        self._action_space_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
    def reset(self) -> np.ndarray:
        self.state = self._solved_state.copy()
        return self.state
        
    def scramble(self, depth: int) -> np.ndarray:
        """
        Scrambles the puzzle by performing random moves from the solved state.
        Ensures the state remains solvable (since random permutations can be unsolvable).
        """
        self.reset()
        last_action = -1
        # Inverse action mapping to avoid immediately undoing the last move
        inverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        
        for _ in range(depth):
            # Find all valid moves
            valid_actions = []
            blank_idx = np.where(self.state == 0)[0][0]
            r, c = blank_idx // self.N, blank_idx % self.N
            
            if r > 0: valid_actions.append(0) # UP
            if r < self.N - 1: valid_actions.append(1) # DOWN
            if c > 0: valid_actions.append(2) # LEFT
            if c < self.N - 1: valid_actions.append(3) # RIGHT
            
            # Try to avoid backtracking
            filtered_actions = [a for a in valid_actions if a != last_action]
            if not filtered_actions:
                filtered_actions = valid_actions
                
            action = random.choice(filtered_actions)
            self.step(action)
            last_action = inverse_actions.get(action, -1)
            
        return self.state
        
    def step(self, action_idx: int) -> float:
        # Locate the blank tile
        blank_idx = np.where(self.state == 0)[0][0]
        r, c = blank_idx // self.N, blank_idx % self.N
        
        # Calculate target position for the blank space
        tr, tc = r, c
        if action_idx == 0: tr -= 1 # UP
        elif action_idx == 1: tr += 1 # DOWN
        elif action_idx == 2: tc -= 1 # LEFT
        elif action_idx == 3: tc += 1 # RIGHT
        
        # If the move is valid, swap tiles
        if 0 <= tr < self.N and 0 <= tc < self.N:
            target_idx = tr * self.N + tc
            self.state[blank_idx], self.state[target_idx] = self.state[target_idx], self.state[blank_idx]
            
        return 1.0 # step cost is 1.0
        
    def get_state(self) -> np.ndarray:
        return self.state.copy()
        
    def set_state(self, state: np.ndarray):
        self.state = np.array(state, dtype=np.int8)
        
    def is_solved(self) -> bool:
        return np.array_equal(self.state, self._solved_state)
        
    @property
    def action_space_names(self) -> list:
        return self._action_space_names
        
    @property
    def state_dim(self) -> int:
        return self.L
        
    @property
    def one_hot_dim(self) -> int:
        return self.L * self.L
        
    def to_one_hot(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            one_hot = np.zeros((self.L, self.L), dtype=np.float32)
            one_hot[np.arange(self.L), state] = 1.0
            return one_hot.flatten()
        else:
            B = state.shape[0]
            one_hot = np.zeros((B, self.L, self.L), dtype=np.float32)
            one_hot[np.arange(B)[:, None], np.arange(self.L), state] = 1.0
            return one_hot.reshape(B, -1)
            
    def get_base_heuristic(self, state: np.ndarray) -> float:
        """
        Computes Manhattan Distance, which is a strictly admissible heuristic.
        """
        dist = 0
        for i in range(self.L):
            val = state[i]
            if val != 0: # Do not include the blank space in Manhattan Distance
                # Current position
                curr_r, curr_c = i // self.N, i % self.N
                # Solved position
                solved_idx = val - 1
                solved_r, solved_c = solved_idx // self.N, solved_idx % self.N
                
                dist += abs(curr_r - solved_r) + abs(curr_c - solved_c)
        return float(dist)
