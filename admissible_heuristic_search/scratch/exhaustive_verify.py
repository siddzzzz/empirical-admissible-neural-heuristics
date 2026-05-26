import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admissible_heuristic_search.envs.lights_out import LightsOut
from admissible_heuristic_search.models.heuristic_net import HeuristicNet
from admissible_heuristic_search.common.solver import AStarSolver

def get_all_states():
    # Generate all 512 binary configurations of size 9
    states = []
    for i in range(512):
        # Convert to binary list of length 9
        binary = [int(x) for x in format(i, '09b')]
        states.append(np.array(binary))
    return states

def exhaustive_bfs(puzzle):
    # Runs BFS to find exact optimal distance for all 512 states
    solved_state = np.zeros(9, dtype=int)
    queue = [(solved_state.copy(), 0)]
    visited = {tuple(solved_state): 0}
    
    while queue:
        state, dist = queue.pop(0)
        
        # Try all 9 actions
        for action in range(9):
            puzzle.set_state(state.copy())
            puzzle.step(action)
            next_state = puzzle.get_state()
            next_tuple = tuple(next_state)
            
            if next_tuple not in visited:
                visited[next_tuple] = dist + 1
                queue.append((next_state.copy(), dist + 1))
                
    return visited

def run_verification():
    device = "cpu"
    puzzle = LightsOut(W=3, H=3)
    
    # 1. Compute exact optimal distances for all states
    optimal_distances = exhaustive_bfs(puzzle)
    print(f"Total reachable states in BFS: {len(optimal_distances)}")
    
    # 2. Load models
    weights_admissible = "trained_models/admissible_lightsout_3x3.pt"
    weights_mse = "trained_models/mse_lightsout_3x3.pt"
    
    model_admissible = HeuristicNet(input_dim=puzzle.one_hot_dim).to(device)
    model_admissible.load_state_dict(torch.load(weights_admissible, map_location=device))
    model_admissible.eval()
    
    model_mse = HeuristicNet(input_dim=puzzle.one_hot_dim).to(device)
    model_mse.load_state_dict(torch.load(weights_mse, map_location=device))
    model_mse.eval()
    
    # Calibrate safety offset for admissible model on validation scrambles
    # (Just like we do in evaluate_all.py)
    # We generate 100 validation scrambles
    max_overest = 0.0
    import random
    random.seed(42)
    for _ in range(100):
        depth = random.randint(1, 8)
        puzzle.scramble(depth)
        state = puzzle.get_state()
        with torch.no_grad():
            one_hot = puzzle.to_one_hot(np.expand_dims(state, 0))
            pred = model_admissible(torch.tensor(one_hot, dtype=torch.float32)).item()
        overest = pred - float(depth)
        if overest > max_overest:
            max_overest = overest
    delta = float(max_overest)
    print(f"Calibrated safety offset (delta): {delta:.4f}")
    
    # 3. Evaluate admissibility for all 512 states
    all_states = get_all_states()
    
    mse_admissible_count = 0
    raw_admissible_count = 0
    calib_admissible_count = 0
    
    for state in all_states:
        state_tuple = tuple(state)
        true_cost = optimal_distances.get(state_tuple, 0.0)
        
        # Skip solved state since h(solved) is forced to 0
        if true_cost == 0:
            mse_admissible_count += 1
            raw_admissible_count += 1
            calib_admissible_count += 1
            continue
            
        with torch.no_grad():
            one_hot = puzzle.to_one_hot(np.expand_dims(state, 0))
            tensor_state = torch.tensor(one_hot, dtype=torch.float32)
            
            pred_mse = model_mse(tensor_state).item()
            pred_raw = model_admissible(tensor_state).item()
            
            # Calibrated prediction
            base_h = puzzle.get_base_heuristic(state)
            pred_calib = max(base_h, pred_raw - delta)
            
        if pred_mse <= true_cost:
            mse_admissible_count += 1
        if pred_raw <= true_cost:
            raw_admissible_count += 1
        if pred_calib <= true_cost:
            calib_admissible_count += 1
            
    print(f"Results over all {len(all_states)} states:")
    print(f"  MSE Heuristic Admissibility:      {mse_admissible_count / 512 * 100:.2f}% ({mse_admissible_count}/512)")
    print(f"  Raw Admissible Heuristic:         {raw_admissible_count / 512 * 100:.2f}% ({raw_admissible_count}/512)")
    print(f"  Calibrated Admissible Heuristic:  {calib_admissible_count / 512 * 100:.2f}% ({calib_admissible_count}/512)")

if __name__ == '__main__':
    run_verification()
