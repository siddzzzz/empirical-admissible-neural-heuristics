import numpy as np
import random
from admissible_heuristic_search.common.env import CombinatorialPuzzle

class Cube2x2(CombinatorialPuzzle):
    """
    2x2 Rubik's Cube environment implementing the CombinatorialPuzzle interface.
    Sticker Representation: 24 integers in [0, 5] representing U, D, F, B, L, R colors.
    """
    def __init__(self, use_macros: bool = True):
        # Canonical solved state
        self._solved_state = np.array([
            0, 0, 0, 0, # U
            1, 1, 1, 1, # D
            2, 2, 2, 2, # F
            3, 3, 3, 3, # B
            4, 4, 4, 4, # L
            5, 5, 5, 5  # R
        ], dtype=np.int8)
        
        self.state = self._solved_state.copy()
        
        # Build moves permutations
        self._moves = {}
        self._action_space_names = []
        self._action_costs = []
        self._define_moves(use_macros)
        
    def reset(self) -> np.ndarray:
        self.state = self._solved_state.copy()
        return self.state
        
    def scramble(self, depth: int) -> np.ndarray:
        self.reset()
        for _ in range(depth):
            # Only scramble using primitive actions (0 to 17)
            act = random.randint(0, 17)
            self.step(act)
        return self.state
        
    def step(self, action_idx: int) -> float:
        action_name = self._action_space_names[action_idx]
        perm = self._moves[action_name]
        self.state = self.state[perm]
        return self._action_costs[action_idx]
        
    def get_state(self) -> np.ndarray:
        return self.state.copy()
        
    def set_state(self, state: np.ndarray):
        self.state = np.array(state, dtype=np.int8)
        
    def is_solved(self) -> bool:
        for i in range(6):
            if not np.all(self.state[i*4:(i+1)*4] == self.state[i*4]):
                return False
        return True
        
    @property
    def action_space_names(self) -> list:
        return self._action_space_names
        
    @property
    def state_dim(self) -> int:
        return 24
        
    @property
    def one_hot_dim(self) -> int:
        return 144
        
    def to_one_hot(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            one_hot = np.zeros((24, 6), dtype=np.float32)
            one_hot[np.arange(24), state] = 1.0
            return one_hot.flatten()
        else:
            B = state.shape[0]
            one_hot = np.zeros((B, 24, 6), dtype=np.float32)
            one_hot[np.arange(B)[:, None], np.arange(24), state] = 1.0
            return one_hot.reshape(B, -1)
            
    def get_base_heuristic(self, state: np.ndarray) -> float:
        # No non-trivial analytical admissible heuristic exists for the Rubik's Cube
        return 0.0
        
    def _define_moves(self, use_macros: bool):
        I = list(range(24))
        
        def make_perm(face_cycle, edge_cycles):
            p = I.copy()
            # Face cycle
            p[face_cycle[0]], p[face_cycle[1]], p[face_cycle[3]], p[face_cycle[2]] = \
                p[face_cycle[2]], p[face_cycle[0]], p[face_cycle[1]], p[face_cycle[3]]
            # Edge cycles
            for c in edge_cycles:
                temp = p[c[3]]
                p[c[3]] = p[c[2]]
                p[c[2]] = p[c[1]]
                p[c[1]] = p[c[0]]
                p[c[0]] = temp
            return p

        # 90-degree Clockwise face rotations
        self._moves['U'] = make_perm([0, 1, 2, 3], [[12, 20, 8, 16], [13, 21, 9, 17]])
        self._moves['D'] = make_perm([4, 5, 6, 7], [[10, 22, 14, 18], [11, 23, 15, 19]])
        self._moves['F'] = make_perm([8, 9, 10, 11], [[2, 20, 5, 19], [3, 22, 4, 17]])
        self._moves['B'] = make_perm([12, 13, 14, 15], [[1, 16, 6, 23], [0, 18, 7, 21]])
        self._moves['L'] = make_perm([16, 17, 18, 19], [[0, 8, 4, 15], [2, 10, 6, 13]])
        self._moves['R'] = make_perm([20, 21, 22, 23], [[3, 12, 7, 9], [1, 14, 5, 11]])

        # Generate Double (2) and Inverse (') moves
        base_moves = list(self._moves.keys())
        for move in base_moves:
            p1 = self._moves[move]
            p2 = [p1[i] for i in p1]
            p3 = [p1[i] for i in p2]
            self._moves[move + '2'] = p2
            self._moves[move + "'"] = p3

        # Add primitive names and costs
        primitives = [
            'U', "U'", 'U2', 'D', "D'", 'D2',
            'F', "F'", 'F2', 'B', "B'", 'B2',
            'L', "L'", 'L2', 'R', "R'", 'R2'
        ]
        for p in primitives:
            self._action_space_names.append(p)
            self._action_costs.append(1.0) # uniform cost 1.0

        # Register macros if enabled
        if use_macros:
            self._register_macro("Sexy", "R U R' U'")
            self._register_macro("Sune", "R U R' U R U2 R'")
            self._register_macro("J-Perm", "R U R' F' R U R' U' R' F R2 U' R' U'")
            self._register_macro("Y-Perm", "F R U' R' U' R U R' F' R U R' U' R' F R F'")

    def _register_macro(self, name: str, move_string: str):
        moves = move_string.split()
        composed_perm = list(range(24))
        for m in moves:
            composed_perm = [composed_perm[i] for i in self._moves[m]]
        self._moves[name] = composed_perm
        self._action_space_names.append(name)
        self._action_costs.append(float(len(moves)))
