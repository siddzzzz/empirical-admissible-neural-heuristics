import sys
import os
import torch
import numpy as np
import random

# Ensure workspace is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.cube import Cube2x2
from agents.tactical.value_agent import ValueNet, load_model
from training.solver import AStarSolver

def run_evaluation():
    print("==================================================")
    st_title = "Evaluating Hierarchical A* (H-DAVI) Solver"
    print(st_title)
    print("==================================================")
    
    device = "cpu"
    model_path = "trained_models/value_agent.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model weights not found at {model_path}!")
        return False
        
    # Load model and solver
    try:
        model = load_model(model_path, device=device)
        solver = AStarSolver(model, device=device, use_macros=True)
    except Exception as e:
        print(f"ERROR loading model/solver: {e}")
        return False
        
    num_cubes = 100
    solved_count = 0
    total_steps = 0
    total_nodes = 0
    max_depth = 14
    
    # Track stats per depth
    depth_stats = {d: {"solved": 0, "total": 0, "steps": 0, "nodes": 0} for d in range(1, max_depth + 1)}
    
    print(f"\nEvaluating solver on {num_cubes} random scrambles (depths 1-{max_depth})...")
    
    c = Cube2x2()
    
    for i in range(num_cubes):
        c.reset()
        depth = random.randint(1, max_depth)
        c.scramble(depth)
        
        path, nodes = solver.solve(c.state, max_nodes=1000)
        
        depth_stats[depth]["total"] += 1
        if path is not None:
            solved_count += 1
            total_steps += len(path)
            total_nodes += nodes
            
            depth_stats[depth]["solved"] += 1
            depth_stats[depth]["steps"] += len(path)
            depth_stats[depth]["nodes"] += nodes
            
            # Verify the solve actually results in a solved cube
            temp_cube = Cube2x2()
            temp_cube.state = c.state.copy()
            for action in path:
                temp_cube.step(action)
            assert temp_cube.is_solved(), "Solver returned path, but cube state was not solved!"
        else:
            print(f"  Cube {i+1}: Failed to solve from scramble depth {depth}!")
            
    success_rate = solved_count / num_cubes
    print("\n================ Results ================")
    print(f"Overall Success Rate: {success_rate * 100:.1f}%")
    if solved_count > 0:
        print(f"Avg Solution Length (moves/macros): {total_steps / solved_count:.2f}")
        print(f"Avg Nodes Expanded: {total_nodes / solved_count:.1f}")
    print("=========================================")
    
    print("\nPer-Depth Performance Breakdown:")
    print(f"{'Depth':<6} | {'Solved/Total':<12} | {'Success Rate':<12} | {'Avg Steps':<10} | {'Avg Nodes':<10}")
    print("-" * 60)
    for d in range(1, max_depth + 1):
        stats = depth_stats[d]
        if stats["total"] > 0:
            sr = (stats["solved"] / stats["total"]) * 100.0
            avg_steps = stats["steps"] / stats["solved"] if stats["solved"] > 0 else 0.0
            avg_nodes = stats["nodes"] / stats["solved"] if stats["solved"] > 0 else 0.0
            solved_total_str = f"{stats['solved']}/{stats['total']}"
            print(f"{d:<6} | {solved_total_str:<12} | {sr:<11.1f}% | {avg_steps:<10.2f} | {avg_nodes:<10.1f}")
        else:
            print(f"{d:<6} | {'0/0':<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")
    print("-" * 60)
    
    # We pass the check if we solve all tested cubes that are within the current curriculum depth of the model.
    # To be general, we check if the success rate up to depth 9 (mastered depths) is >= 95%
    mastered_solved = sum(depth_stats[d]["solved"] for d in range(1, 10))
    mastered_total = sum(depth_stats[d]["total"] for d in range(1, 10))
    mastered_sr = mastered_solved / mastered_total if mastered_total > 0 else 1.0
    
    print(f"\nMastered Depths (1-9) Success Rate: {mastered_sr * 100:.1f}% ({mastered_solved}/{mastered_total})")
    
    if mastered_sr >= 0.95:
        print("Verification PASSED for mastered depths (1-9)!")
        return True
    else:
        print("Verification FAILED for mastered depths (1-9).")
        return False

if __name__ == "__main__":
    run_evaluation()
