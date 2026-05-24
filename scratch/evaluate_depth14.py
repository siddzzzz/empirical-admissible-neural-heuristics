import sys
import os
import torch
import numpy as np
import random

# Ensure workspace is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cube import Cube2x2
from agents.tactical.value_agent import ValueNet, load_model
from training.solver import AStarSolver

def evaluate_at_budget(model, device, depth, num_cubes=50, max_nodes=1000):
    solver = AStarSolver(model, device=device, use_macros=True)
    solved_count = 0
    total_nodes = 0
    c = Cube2x2()
    for _ in range(num_cubes):
        c.reset()
        c.scramble(depth)
        path, nodes = solver.solve(c.state, max_nodes=max_nodes)
        if path is not None:
            solved_count += 1
            total_nodes += nodes
    avg_nodes = total_nodes / solved_count if solved_count > 0 else 0
    return solved_count / num_cubes, avg_nodes

def main():
    device = "cpu"
    model_path = "trained_models/value_agent.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}!")
        return
        
    model = load_model(model_path, device=device)
    
    print("Evaluating success rate on Depth 14 scrambles with different A* search budgets:")
    for budget in [1000, 2000, 5000]:
        sr, avg_n = evaluate_at_budget(model, device, depth=14, num_cubes=50, max_nodes=budget)
        print(f"  Max Nodes: {budget:<5} | Success Rate: {sr*100:.1f}% | Avg Expanded Nodes (for solves): {avg_n:.1f}")

if __name__ == '__main__':
    main()
