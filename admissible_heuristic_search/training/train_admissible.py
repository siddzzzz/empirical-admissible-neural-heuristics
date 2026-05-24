import sys
import os
import torch
import numpy as np
import random
import time
import argparse
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admissible_heuristic_search.common.env import CombinatorialPuzzle
from admissible_heuristic_search.common.solver import AStarSolver
from admissible_heuristic_search.envs.cube2x2 import Cube2x2
from admissible_heuristic_search.envs.tile_puzzle import TilePuzzle
from admissible_heuristic_search.envs.lights_out import LightsOut
from admissible_heuristic_search.models.heuristic_net import HeuristicNet, AsymmetricLoss

def generate_scramble_batch(puzzle: CombinatorialPuzzle, batch_size: int, max_depth: int) -> np.ndarray:
    states = []
    for _ in range(batch_size):
        d = random.randint(1, max_depth)
        puzzle.scramble(d)
        states.append(puzzle.get_state())
    return np.array(states)

def get_next_states_batch(puzzle: CombinatorialPuzzle, states: np.ndarray):
    B = states.shape[0]
    A = puzzle.num_actions
    D = puzzle.state_dim
    
    next_states = np.zeros((B, A, D), dtype=states.dtype)
    costs = np.zeros((B, A), dtype=np.float32)
    
    for i in range(B):
        state_array = states[i]
        for action in range(A):
            puzzle.set_state(state_array)
            cost = puzzle.step(action)
            next_states[i, action, :] = puzzle.get_state()
            costs[i, action] = cost
            
    return next_states, costs

def is_solved_batch(puzzle: CombinatorialPuzzle, states: np.ndarray) -> np.ndarray:
    B = states.shape[0]
    solved_mask = np.zeros(B, dtype=bool)
    for i in range(B):
        puzzle.set_state(states[i])
        solved_mask[i] = puzzle.is_solved()
    return solved_mask

def evaluate_model(puzzle: CombinatorialPuzzle, model: torch.nn.Module, device: str, 
                   depth: int, num_cubes: int = 30, max_nodes: int = 1000) -> tuple:
    """
    Evaluates the model's solve success rate and admissibility rate.
    """
    model.eval()
    solver = AStarSolver(puzzle, model, device=device)
    solved_count = 0
    admissible_count = 0
    
    test_puzzle = puzzle
    
    for _ in range(num_cubes):
        test_puzzle.reset()
        test_puzzle.scramble(depth)
        state = test_puzzle.get_state()
        
        # Check model prediction admissibility: h(s) <= depth (scramble depth is upper bound of true cost)
        with torch.no_grad():
            one_hot = test_puzzle.to_one_hot(np.expand_dims(state, 0))
            tensor_state = torch.tensor(one_hot, dtype=torch.float32, device=device)
            pred = model(tensor_state).item()
            
        if pred <= float(depth):
            admissible_count += 1
            
        path, _ = solver.solve(state, max_nodes=max_nodes)
        if path is not None:
            solved_count += 1
            
    return solved_count / num_cubes, admissible_count / num_cubes

def train():
    parser = argparse.ArgumentParser(description="Train Provably Admissible Neural Heuristics.")
    parser.add_argument("--puzzle", type=str, default="lightsout", 
                        choices=["cube2x2", "tile8", "lightsout"],
                        help="Puzzle environment to train on.")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="Grid size for Lights Out (3 or 5).")
    parser.add_argument("--alpha", type=float, default=100.0,
                        help="Asymmetric loss penalty multiplier for overestimation.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Safety discount parameter for the Admissible Bellman Operator.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Optimizer learning rate.")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps (default 5000 for verification).")
    
    # Allow running in script mode with default values if args are empty
    args, unknown = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training parameters: Puzzle={args.puzzle}, Alpha={args.alpha}, Epsilon={args.epsilon}, LR={args.lr}")
    
    # Initialize puzzle environment
    if args.puzzle == "cube2x2":
        puzzle = Cube2x2(use_macros=False) # Train on primitives for clean heuristics
        max_depth = 14
    elif args.puzzle == "tile8":
        puzzle = TilePuzzle(N=3)
        max_depth = 31
    elif args.puzzle == "lightsout":
        puzzle = LightsOut(W=args.grid_size, H=args.grid_size)
        max_depth = 15 if args.grid_size == 5 else 10
        
    print(f"Puzzle initialized: State Dim={puzzle.state_dim}, One-Hot Dim={puzzle.one_hot_dim}, Actions={puzzle.num_actions}")
    
    # Initialize model, target model, optimizer, and asymmetric loss
    model = HeuristicNet(input_dim=puzzle.one_hot_dim).to(device)
    target_model = HeuristicNet(input_dim=puzzle.one_hot_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = AsymmetricLoss(alpha=args.alpha)
    
    # Training configurations
    batch_size = 128
    target_update_freq = 50
    eval_freq = 500
    success_threshold = 0.90
    curriculum_depth = 1
    
    # Setup weights save path
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    if args.puzzle == "lightsout":
        model_path = os.path.join(model_dir, f"admissible_lightsout_{args.grid_size}x{args.grid_size}.pt")
    else:
        model_path = os.path.join(model_dir, f"admissible_{args.puzzle}.pt")
    
    step = 0
    start_time = time.time()
    
    print("Starting Admissible Neural Heuristic training loop...")
    print(f"Initial Scramble Depth: {curriculum_depth}")
    
    try:
        while step < args.steps and curriculum_depth <= max_depth:
            model.train()
            
            # 1. Generate scrambled states
            states = generate_scramble_batch(puzzle, batch_size, curriculum_depth)
            
            # 2. Get next-states for all valid actions
            next_states, costs = get_next_states_batch(puzzle, states) # shapes: (B, A, D), (B, A)
            B, A, D = next_states.shape
            next_states_flat = next_states.reshape(-1, D) # (B * A, D)
            
            # 3. Check which next-states are solved
            solved_mask = is_solved_batch(puzzle, next_states_flat) # (B * A,) boolean array
            
            # 4. Predict target values for next states
            one_hot_next = puzzle.to_one_hot(next_states_flat)
            next_states_tensor = torch.tensor(one_hot_next, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                target_values = target_model(next_states_tensor).squeeze(-1) # (B * A,)
                
                # Solved states have remaining cost of 0.0
                target_values[solved_mask] = 0.0
                
                # Add step costs
                costs_tensor = torch.tensor(costs.flatten(), dtype=torch.float32, device=device)
                target_values = target_values + costs_tensor
                
                # Get minimum cost-to-go target
                target_values = target_values.view(B, A)
                y = torch.min(target_values, dim=1).values # shape: (B,)
                
                # Apply Admissible Bellman Operator: y = max(h0(s), y - epsilon)
                base_h = torch.tensor([puzzle.get_base_heuristic(s) for s in states], dtype=torch.float32, device=device)
                y = torch.max(base_h, y - args.epsilon)
                
            # 5. Predict values for current states and update network
            one_hot_states = puzzle.to_one_hot(states)
            states_tensor = torch.tensor(one_hot_states, dtype=torch.float32, device=device)
            predictions = model(states_tensor).squeeze(-1) # (B,)
            
            # Target of solved states is strictly 0.0
            states_solved_mask = is_solved_batch(puzzle, states)
            y[states_solved_mask] = 0.0
            
            loss = loss_fn(predictions, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # 6. Update target network
            if step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # 7. Evaluate and adjust curriculum
            if step % eval_freq == 0:
                # Dynamically scale evaluation budget with curriculum depth to prevent getting stuck
                if curriculum_depth == max_depth:
                    budget = 10000
                elif curriculum_depth >= 11:
                    budget = 6000
                elif curriculum_depth >= 8:
                    budget = 4000
                elif curriculum_depth >= 5:
                    budget = 2500
                else:
                    budget = 1200
                val_sr, admissible_rate = evaluate_model(
                    puzzle, model, device, curriculum_depth, num_cubes=25, max_nodes=budget
                )
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                elapsed = time.time() - start_time
                print(f"Step {step:<5} | Elapsed: {elapsed:.1f}s | Depth: {curriculum_depth:<2} | "
                      f"Loss: {loss.item():.4f} | Solve Rate: {val_sr*100:.1f}% | "
                      f"Raw Admissibility: {admissible_rate*100:.1f}%")
                
                if val_sr >= success_threshold:
                    print(f"--- Curriculum Level Up! Passed Depth {curriculum_depth} with {val_sr*100:.1f}% success rate. ---")
                    curriculum_depth += 1
                    if curriculum_depth > max_depth:
                        print("🎉 CONGRATULATIONS! Admissible Neural Heuristic training successfully completed curriculum! 🎉")
                        break
                    
                    # Save checkpoint
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved checkpoint to {model_path}")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Final weights saved to {model_path}")

if __name__ == '__main__':
    train()
