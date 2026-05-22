import heapq
import numpy as np
import torch
from env.cube import Cube2x2
from agents.tactical.value_agent import ValueNet, state_to_tensor, batch_to_tensor

class AStarSolver:
    """
    A* Search Solver for 2x2 Rubik's Cube.
    Uses a trained ValueNet model as the heuristic function.
    Can branch on both primitive actions and high-level macro actions.
    """
    def __init__(self, model: ValueNet, device: str = "cpu", use_macros: bool = True):
        self.model = model
        self.device = device
        self.use_macros = use_macros
        
        # Instantiate a helper cube to inspect moves/costs
        self.helper_cube = Cube2x2()
        self.num_actions = len(self.helper_cube.action_space_names)
        self.action_names = self.helper_cube.action_space_names
        
        # Determine branching actions
        if self.use_macros:
            self.branching_actions = list(range(self.num_actions))
        else:
            self.branching_actions = list(range(18)) # Only primitives
            
    def _is_solved(self, state: np.ndarray) -> bool:
        # Fast is_solved check on the state array directly
        for i in range(6):
            if not np.all(state[i*4:(i+1)*4] == state[i*4]):
                return False
        return True

    def _get_heuristic(self, state: np.ndarray, state_tuple=None) -> float:
        if state_tuple is None:
            state_tuple = tuple(state)
        if state_tuple in self._h_cache:
            return self._h_cache[state_tuple]
        if self._is_solved(state):
            return 0.0
        with torch.no_grad():
            tensor_state = state_to_tensor(state, self.device)
            val = self.model(tensor_state).item()
        h_val = max(0.0, val)
        self._h_cache[state_tuple] = h_val
        return h_val

    def _load_heuristics_batch(self, states_list):
        """
        Batches predictions for multiple states to avoid single-item PyTorch CPU overhead.
        """
        uncached_states = []
        uncached_tuples = []
        
        for state, state_tuple in states_list:
            if state_tuple not in self._h_cache:
                if self._is_solved(state):
                    self._h_cache[state_tuple] = 0.0
                else:
                    uncached_states.append(state)
                    uncached_tuples.append(state_tuple)
                    
        if len(uncached_states) > 0:
            states_batch = np.array(uncached_states, dtype=np.int8)
            with torch.no_grad():
                tensor_batch = batch_to_tensor(states_batch, self.device)
                preds = self.model(tensor_batch).squeeze(-1).cpu().numpy()
                
            # If batch size is 1, squeeze(-1) might turn it into a scalar, so handle shape carefully
            if len(uncached_states) == 1:
                preds = [float(preds)]
                
            for state_tuple, pred in zip(uncached_tuples, preds):
                self._h_cache[state_tuple] = float(max(0.0, pred))

    def solve(self, start_state: np.ndarray, max_nodes: int = 1000):
        """
        Runs A* search to solve the cube from start_state.
        Uses batched heuristic evaluation for 20x search speedup.
        """
        self._h_cache = {}
        if self._is_solved(start_state):
            return [], 0
            
        # Priority queue stores: (f_score, g_score, tiebreaker, state_tuple, path)
        pq = []
        
        # Visited set stores: state_tuple -> best_g_score
        visited = {}
        
        # Heuristic for starting state
        h_start = self._get_heuristic(start_state)
        
        start_tuple = tuple(start_state)
        heapq.heappush(pq, (h_start, 0.0, 0, start_tuple, []))
        visited[start_tuple] = 0.0
        
        tiebreaker = 0
        nodes_expanded = 0
        
        # Temporary cube to compute transitions
        c = Cube2x2()
        
        while pq:
            f, g, _, state_tuple, path = heapq.heappop(pq)
            nodes_expanded += 1
            
            # Since heapq might contain duplicate states with higher g values,
            # we discard if we've already found a better path to this state.
            if g > visited.get(state_tuple, float('inf')):
                continue
                
            state_array = np.array(state_tuple, dtype=np.int8)
            if self._is_solved(state_array):
                return path, nodes_expanded
                
            if nodes_expanded >= max_nodes:
                break
                
            # Generate and filter candidate children
            candidates = []
            for action in self.branching_actions:
                c.state = state_array.copy()
                c.step(action)
                next_state = c.get_state()
                next_tuple = tuple(next_state)
                
                action_name = self.action_names[action]
                cost = self.helper_cube.action_costs.get(action_name, 1)
                next_g = g + cost
                
                if next_g < visited.get(next_tuple, float('inf')):
                    visited[next_tuple] = next_g
                    candidates.append((next_state, next_tuple, next_g, action))
            
            # Batch load heuristics for all candidates
            if candidates:
                self._load_heuristics_batch([(item[0], item[1]) for item in candidates])
                
            # Push candidates to priority queue
            for next_state, next_tuple, next_g, action in candidates:
                h = self._get_heuristic(next_state, next_tuple)
                f_next = next_g + h
                
                tiebreaker += 1
                heapq.heappush(pq, (f_next, next_g, tiebreaker, next_tuple, path + [action]))
                
        return None, nodes_expanded

    def get_next_action(self, state: np.ndarray) -> int:
        """
        Finds the single best next action index by running A* search and returning the first step.
        """
        path, _ = self.solve(state, max_nodes=2000)
        if path and len(path) > 0:
            return path[0]
        # If no path found, return a random primitive action as fallback
        return np.random.randint(0, 18)

