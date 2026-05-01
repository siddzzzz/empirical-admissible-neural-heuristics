import numpy as np
import random

class Cube2x2:
    """
    A 2x2 Rubik's Cube environment.
    State is represented as a 1D numpy array of 24 integers (0-5).
    Colors:
    0: White (U)
    1: Yellow (D)
    2: Green (F)
    3: Blue (B)
    4: Orange (L)
    5: Red (R)

    Faces:
    U: 0, 1, 2, 3
    D: 4, 5, 6, 7
    F: 8, 9, 10, 11
    B: 12, 13, 14, 15
    L: 16, 17, 18, 19
    R: 20, 21, 22, 23
    """
    
    def __init__(self):
        self.reset()
        
        # Define base permutations for the 6 faces (Clockwise 90 degrees)
        self._moves = {}
        self._define_moves()
        
    def reset(self):
        # 6 colors, 4 stickers each = 24 stickers
        self.state = np.array([
            0, 0, 0, 0, # U
            1, 1, 1, 1, # D
            2, 2, 2, 2, # F
            3, 3, 3, 3, # B
            4, 4, 4, 4, # L
            5, 5, 5, 5  # R
        ], dtype=np.int8)
        
    def is_solved(self):
        # Check if each face has all the same colors
        for i in range(6):
            if not np.all(self.state[i*4:(i+1)*4] == self.state[i*4]):
                return False
        return True

    def get_state(self):
        return self.state.copy()

    def _define_moves(self):
        # Identity permutation
        I = list(range(24))
        
        # Helper to create a permutation
        def make_perm(face_cycle, edge_cycles):
            p = I.copy()
            # Face rotation: 0->1, 1->3, 3->2, 2->0 (indices within the face block)
            # which maps to indices: face_start + [0, 1, 3, 2]
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

        # We define U, D, F, B, L, R (90 degree clockwise)
        self._moves['U'] = make_perm([0, 1, 2, 3], [[12, 20, 8, 16], [13, 21, 9, 17]])
        self._moves['D'] = make_perm([4, 5, 6, 7], [[10, 22, 14, 18], [11, 23, 15, 19]])
        self._moves['F'] = make_perm([8, 9, 10, 11], [[2, 20, 5, 19], [3, 22, 4, 17]])
        self._moves['B'] = make_perm([12, 13, 14, 15], [[1, 16, 6, 23], [0, 18, 7, 21]])
        self._moves['L'] = make_perm([16, 17, 18, 19], [[0, 8, 4, 15], [2, 10, 6, 13]])
        self._moves['R'] = make_perm([20, 21, 22, 23], [[3, 12, 7, 9], [1, 14, 5, 11]])

        # Create prime (') and double (2) moves
        base_moves = list(self._moves.keys())
        for move in base_moves:
            p1 = self._moves[move]
            p2 = [p1[i] for i in p1]
            p3 = [p1[i] for i in p2]
            
            self._moves[move + '2'] = p2
            self._moves[move + "'"] = p3

        self.action_space_names = [
            'U', "U'", 'U2',
            'D', "D'", 'D2',
            'F', "F'", 'F2',
            'B', "B'", 'B2',
            'L', "L'", 'L2',
            'R', "R'", 'R2'
        ]
        
        # Track cost (number of base moves) for each action
        self.action_costs = {name: 1 for name in self.action_space_names}
        
        # Register standard human macros
        self._register_macro("Sexy", "R U R' U'")
        self._register_macro("Sune", "R U R' U R U2 R'")
        self._register_macro("J-Perm", "R U R' F' R U R' U' R' F R2 U' R' U'")
        self._register_macro("Y-Perm", "F R U' R' U' R U R' F' R U R' U' R' F R F'")

    def _register_macro(self, name, move_string):
        """
        Compiles a space-separated sequence of moves into a single permutation
        and registers it as a new action in the action space.
        """
        moves = move_string.split()
        
        # Start with the identity permutation
        composed_perm = list(range(24))
        
        for move in moves:
            if move not in self._moves:
                raise ValueError(f"Unknown base move {move} in macro {name}")
            perm = self._moves[move]
            # Compose permutation: P_new = P_old[perm]
            composed_perm = [composed_perm[i] for i in perm]
            
        self._moves[name] = composed_perm
        self.action_space_names.append(name)
        self.action_costs[name] = len(moves)
        
    def step(self, action_idx):
        action_name = self.action_space_names[action_idx]
        perm = self._moves[action_name]
        self.state = self.state[perm]
        return self.state

    def scramble(self, n_moves=20):
        # Scramble using only primitive moves (indices 0 to 17)
        for _ in range(n_moves):
            action_idx = random.randint(0, 17)
            self.step(action_idx)
