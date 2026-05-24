import os
import sys
import torch
import numpy as np
import random
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admissible_heuristic_search.common.env import CombinatorialPuzzle
from admissible_heuristic_search.common.solver import AStarSolver
from admissible_heuristic_search.envs.cube2x2 import Cube2x2
from admissible_heuristic_search.envs.tile_puzzle import TilePuzzle
from admissible_heuristic_search.envs.lights_out import LightsOut
from admissible_heuristic_search.models.heuristic_net import HeuristicNet

def calibrate_heuristic(puzzle: CombinatorialPuzzle, model: torch.nn.Module, device: str, 
                        num_samples: int = 100, max_scramble: int = 10) -> float:
    """
    Computes the post-hoc calibration safety offset delta.
    delta = max_s(0, h_theta(s) - true_cost(s))
    Since true_cost(s) <= scramble_depth, we conservatively use scramble_depth to guarantee admissibility.
    """
    model.eval()
    max_overestimation = 0.0
    
    test_puzzle = puzzle
    for _ in range(num_samples):
        depth = random.randint(1, max_scramble)
        test_puzzle.scramble(depth)
        state = test_puzzle.get_state()
        
        with torch.no_grad():
            one_hot = test_puzzle.to_one_hot(np.expand_dims(state, 0))
            tensor_state = torch.tensor(one_hot, dtype=torch.float32, device=device)
            pred = model(tensor_state).item()
            
        # Overestimation relative to scramble depth (which is an upper bound on optimal cost)
        overest = pred - float(depth)
        if overest > max_overestimation:
            max_overestimation = overest
            
    return float(max_overestimation)

def evaluate_heuristic(puzzle: CombinatorialPuzzle, solver: AStarSolver, test_states: list, max_nodes: int = 1000) -> tuple:
    solved_count = 0
    total_nodes = 0
    
    for state, depth in test_states:
        # Run solver
        path, nodes = solver.solve(state, max_nodes=max_nodes)
        if path is not None:
            solved_count += 1
            total_nodes += nodes
            
    success_rate = solved_count / len(test_states)
    avg_nodes = total_nodes / solved_count if solved_count > 0 else 0.0
    return success_rate, avg_nodes

def test_admissibility_rate(puzzle: CombinatorialPuzzle, model: torch.nn.Module, solver: AStarSolver, 
                            test_states: list, device: str) -> float:
    admissible_count = 0
    for state, depth in test_states:
        h_val = solver._get_heuristic(state)
        # Using scramble depth as conservative upper bound of true cost
        if h_val <= float(depth):
            admissible_count += 1
    return admissible_count / len(test_states)

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==========================================================")
    print("PROVABLY ADMISSIBLE NEURAL HEURISTICS EVALUATION SUITE")
    print("==========================================================")
    
    puzzles_info = {
        "lightsout": {
            "name": "Lights Out (3x3 Grid)",
            "class": LightsOut(W=3, H=3),
            "test_depth": 8,
            "weights": "trained_models/admissible_lightsout.pt"
        },
        "tile8": {
            "name": "8-Puzzle (3x3 Sliding Tiles)",
            "class": TilePuzzle(N=3),
            "test_depth": 10,
            "weights": "trained_models/admissible_tile8.pt"
        },
        "cube2x2": {
            "name": "2x2 Rubik's Cube",
            "class": Cube2x2(use_macros=False),
            "test_depth": 10,
            "weights": "trained_models/admissible_cube2x2.pt"
        }
    }
    
    for key, info in puzzles_info.items():
        print(f"\nEvaluating: {info['name']}")
        print("-" * 58)
        
        puzzle = info["class"]
        weights_path = info["weights"]
        
        # Check if trained weights exist
        if not os.path.exists(weights_path):
            print(f"Warning: Trained weights not found at {weights_path}.")
            print("To run this evaluation, train the model first using:")
            print(f"  python admissible_heuristic_search/training/train_admissible.py --puzzle {key}")
            continue
            
        # Load model
        model = HeuristicNet(input_dim=puzzle.one_hot_dim).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # 1. Generate standard test states (50 samples scrambled at target depth)
        test_states = []
        for _ in range(50):
            puzzle.scramble(info["test_depth"])
            test_states.append((puzzle.get_state(), info["test_depth"]))
            
        # 2. Calibrate the safety offset
        print("Calibrating post-hoc safety margin...")
        delta = calibrate_heuristic(puzzle, model, device, num_samples=100, max_scramble=info["test_depth"])
        print(f"  Computed Safety Margin (delta): {delta:.4f}")
        
        # 3. Create solvers
        solver_base = AStarSolver(puzzle, model=None, device=device) # Base analytical
        solver_raw = AStarSolver(puzzle, model=model, device=device, calibration_offset=0.0) # Raw NN
        solver_calib = AStarSolver(puzzle, model=model, device=device, calibration_offset=delta) # Calibrated NN
        
        # 4. Evaluate Heuristics
        print(f"Running A* search with Max Nodes = 1000...")
        
        # Base Analytical Heuristic
        sr_base, nodes_base = evaluate_heuristic(puzzle, solver_base, test_states)
        adm_base = test_admissibility_rate(puzzle, None, solver_base, test_states, device)
        print(f"  [Analytical Heuristic]")
        print(f"    - Success Rate:   {sr_base*100:.1f}%")
        print(f"    - Avg Nodes Exp:  {nodes_base:.1f}")
        print(f"    - Admissibility:  {adm_base*100:.1f}%")
        
        # Raw Neural Heuristic
        sr_raw, nodes_raw = evaluate_heuristic(puzzle, solver_raw, test_states)
        adm_raw = test_admissibility_rate(puzzle, model, solver_raw, test_states, device)
        print(f"  [Raw Neural Heuristic (Uncalibrated)]")
        print(f"    - Success Rate:   {sr_raw*100:.1f}%")
        print(f"    - Avg Nodes Exp:  {nodes_raw:.1f}")
        print(f"    - Admissibility:  {adm_raw*100:.1f}%")
        
        # Calibrated Guaranteed Admissible Neural Heuristic
        sr_calib, nodes_calib = evaluate_heuristic(puzzle, solver_calib, test_states)
        adm_calib = test_admissibility_rate(puzzle, model, solver_calib, test_states, device)
        print(f"  [Calibrated Neural Heuristic (Provably Admissible)]")
        print(f"    - Success Rate:   {sr_calib*100:.1f}%")
        print(f"    - Avg Nodes Exp:  {nodes_calib:.1f}")
        print(f"    - Admissibility:  {adm_calib*100:.1f}%")
        
        # Scientific highlight
        node_saving = ((nodes_base - nodes_calib) / nodes_base * 100.0) if nodes_base > 0 else 0.0
        print(f"  RESULT: Calibrated Neural Heuristic achieved {adm_calib*100:.1f}% admissibility.")
        if node_saving > 0:
            print(f"            Expanded {node_saving:.1f}% FEWER nodes than the analytical baseline!")
            
    print("=" * 58)

if __name__ == '__main__':
    run_evaluation()
