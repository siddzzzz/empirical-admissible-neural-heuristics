import heapq
import numpy as np
import torch
from admissible_heuristic_search.common.env import CombinatorialPuzzle

class AStarSolver:
    """
    Generalized A* Search Solver for combinatorial puzzles.
    Uses a trained heuristic model with optional admissibility calibration.
    Supports batched heuristic evaluations for extreme speedups.
    """
    def __init__(self, puzzle: CombinatorialPuzzle, model: torch.nn.Module = None, 
                 device: str = "cpu", calibration_offset: float = 0.0):
        self.puzzle = puzzle
        self.model = model
        self.device = device
        self.calibration_offset = calibration_offset
        self._h_cache = {}
        
    def _is_solved(self, state: np.ndarray) -> bool:
        # Save and restore puzzle state to check solvedness cleanly
        current_state = self.puzzle.get_state()
        self.puzzle.set_state(state)
        solved = self.puzzle.is_solved()
        self.puzzle.set_state(current_state)
        return solved

    def _get_heuristic(self, state: np.ndarray, state_tuple=None) -> float:
        if state_tuple is None:
            state_tuple = tuple(state)
        if state_tuple in self._h_cache:
            return self._h_cache[state_tuple]
            
        base_h = self.puzzle.get_base_heuristic(state)
        if self._is_solved(state):
            return 0.0
            
        if self.model is None:
            # Fall back to base analytical heuristic if no model is loaded
            return base_h
            
        with torch.no_grad():
            one_hot = self.puzzle.to_one_hot(np.expand_dims(state, 0)) # shape: (1, one_hot_dim)
            tensor_state = torch.tensor(one_hot, dtype=torch.float32, device=self.device)
            val = self.model(tensor_state).item()
            
        # Calibrated heuristic: max(base_heuristic, model_prediction - offset)
        h_val = float(max(base_h, val - self.calibration_offset))
        self._h_cache[state_tuple] = h_val
        return h_val

    def _load_heuristics_batch(self, states_list):
        """
        Batches predictions for multiple states to avoid single-item PyTorch CPU overhead.
        """
        if self.model is None:
            return # No neural heuristic to batch-load
            
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
            states_batch = np.array(uncached_states)
            one_hot_batch = self.puzzle.to_one_hot(states_batch) # shape: (B, one_hot_dim)
            
            with torch.no_grad():
                tensor_batch = torch.tensor(one_hot_batch, dtype=torch.float32, device=self.device)
                preds = self.model(tensor_batch).squeeze(-1).cpu().numpy()
                
            # If batch size is 1, squeeze(-1) makes it a scalar; handle shape carefully
            if len(uncached_states) == 1:
                preds = [float(preds)]
                
            for state_tuple, pred, state in zip(uncached_tuples, preds, uncached_states):
                base_h = self.puzzle.get_base_heuristic(state)
                # Calibrated heuristic: max(base_heuristic, model_prediction - offset)
                self._h_cache[state_tuple] = float(max(base_h, pred - self.calibration_offset))

    def solve(self, start_state: np.ndarray, max_nodes: int = 1000):
        """
        Runs A* search to solve the puzzle from start_state.
        Returns:
            path: list of action indexes, or None if failed
            nodes_expanded: number of nodes expanded
        """
        self._h_cache = {}
        self.reopen_count = 0
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
        
        # Track unique expanded states to count reopenings
        expanded_states = set()
        
        # Helper puzzle instance to compute transitions
        # We assume get_state/set_state restores state correctly
        puzzle_helper = self.puzzle
        
        while pq:
            f, g, _, state_tuple, path = heapq.heappop(pq)
            
            if g > visited.get(state_tuple, float('inf')):
                continue
                
            nodes_expanded += 1
            if state_tuple in expanded_states:
                self.reopen_count += 1
            else:
                expanded_states.add(state_tuple)
                
            state_array = np.array(state_tuple)
            if self._is_solved(state_array):
                return path, nodes_expanded
                
            if nodes_expanded >= max_nodes:
                break
                
            # Generate and filter candidate children
            candidates = []
            for action in range(self.puzzle.num_actions):
                puzzle_helper.set_state(state_array)
                cost = puzzle_helper.step(action)
                next_state = puzzle_helper.get_state()
                next_tuple = tuple(next_state)
                next_g = g + cost
                
                if next_g < visited.get(next_tuple, float('inf')):
                    visited[next_tuple] = next_g
                    candidates.append((next_state.copy(), next_tuple, next_g, action))
            
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
