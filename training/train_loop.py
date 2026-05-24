import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
import time
import gc

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cube import Cube2x2
from agents.tactical.value_agent import ValueNet, batch_to_tensor, save_model
from training.solver import AStarSolver

def is_solved_batch(states: np.ndarray) -> np.ndarray:
    """
    Checks if a batch of states are in a solved state.
    states shape: (N, 24)
    returns: boolean array of shape (N,)
    """
    reshaped = states.reshape(-1, 6, 4)
    return np.all(reshaped[:, :, 1:] == reshaped[:, :, :1], axis=(1, 2))

def get_next_states(states: np.ndarray, action_perms: list) -> np.ndarray:
    """
    Applies all 18 primitive actions to a batch of states.
    states shape: (B, 24)
    returns: next_states of shape (B, 18, 24)
    """
    B = states.shape[0]
    next_states = np.zeros((B, 18, 24), dtype=states.dtype)
    for a in range(18):
        next_states[:, a, :] = states[:, action_perms[a]]
    return next_states

def generate_batch(batch_size, max_depth, action_perms):
    """
    Generates a batch of scrambled states, with scramble depths sampled uniformly from 1 to max_depth.
    Uses cached permutations for extreme speed.
    """
    states = np.zeros((batch_size, 24), dtype=np.int8)
    solved_state = np.array([
        0, 0, 0, 0, # U
        1, 1, 1, 1, # D
        2, 2, 2, 2, # F
        3, 3, 3, 3, # B
        4, 4, 4, 4, # L
        5, 5, 5, 5  # R
    ], dtype=np.int8)
    
    for i in range(batch_size):
        state = solved_state.copy()
        # Sample depth uniformly from 1 to max_depth
        d = random.randint(1, max_depth)
        for _ in range(d):
            action_idx = random.randint(0, 17)
            state = state[action_perms[action_idx]]
        states[i] = state
    return states

def evaluate_model(model, device, depth, num_cubes=100, max_nodes=1000) -> float:
    """
    Evaluates the model's solve success rate on random scrambles at the given depth using A* search.
    """
    model.eval()
    solver = AStarSolver(model, device=device, use_macros=True)
    solved_count = 0
    c = Cube2x2()
    for _ in range(num_cubes):
        c.reset()
        c.scramble(depth)
        path, nodes = solver.solve(c.state, max_nodes=max_nodes)
        if path is not None:
            solved_count += 1
    return solved_count / num_cubes

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models
    model = ValueNet().to(device)
    target_model = ValueNet().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.HuberLoss()
    
    # Cache action permutations for fast transitions
    helper_cube = Cube2x2()
    action_perms = [helper_cube._moves[helper_cube.action_space_names[a]] for a in range(18)]
    
    # Curriculum settings
    max_depth = 14
    success_threshold = 0.90 # Require 90% solve rate to advance depth
    curriculum_depth = 1

    # Check if a model already exists to resume training
    model_dir = "trained_models"
    model_path = os.path.join(model_dir, "value_agent.pt")
    if os.path.exists(model_path):
        try:
            print(f"Loading checkpoint from {model_path} to resume training...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            target_model.load_state_dict(model.state_dict())
            print("Successfully loaded model!")
            
            # Find the correct curriculum depth to start with
            print("Evaluating loaded model to determine curriculum starting depth...")
            for d in range(1, max_depth + 1):
                # Use progressive budget for deep scrambles
                budget = 5000 if d == 14 else (3000 if d >= 11 else 1000)
                val_sr = evaluate_model(model, device, d, num_cubes=30, max_nodes=budget)
                print(f"Depth {d} Success Rate: {val_sr*100:.1f}% (A* max nodes: {budget})")
                if val_sr >= success_threshold:
                    curriculum_depth = d + 1
                else:
                    curriculum_depth = max(1, d)
                    break
            curriculum_depth = min(curriculum_depth, max_depth)
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")
            
        # Clean up memory after initial evaluation runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    batch_size = 256
    target_update_freq = 100
    eval_freq = 500
    
    step = 0
    start_time = time.time()
    
    print("Starting Hierarchical Deep Approximate Value Iteration (H-DAVI) Training Loop...")
    print(f"Initial Curriculum Scramble Depth: {curriculum_depth}")
    
    try:
        while curriculum_depth <= max_depth:
            model.train()
            
            # 1. Generate scrambled states
            states = generate_batch(batch_size, curriculum_depth, action_perms)
            
            # 2. Get next-states for all 18 primitive actions
            next_states = get_next_states(states, action_perms) # (B, 18, 24)
            next_states_flat = next_states.reshape(-1, 24) # (B * 18, 24)
            
            # 3. Check which next-states are solved
            solved_mask = is_solved_batch(next_states_flat) # (B * 18,) boolean array
            
            # 4. Predict target values for next states
            next_states_tensor = batch_to_tensor(next_states_flat, device) # (B * 18, 144)
            with torch.no_grad():
                target_values = target_model(next_states_tensor).squeeze(-1) # (B * 18,)
                
                # If a next-state is solved, its remaining distance to solved is 0
                target_values[solved_mask] = 0.0
                
                # Add step cost (1.0 for primitive actions)
                target_values = target_values + 1.0
                
                # Reshape back to (B, 18) and get minimum target cost-to-go
                target_values = target_values.view(batch_size, 18)
                y = torch.min(target_values, dim=1).values # (B,)
                
            # 5. Predict values for current states and update network
            states_tensor = batch_to_tensor(states, device) # (B, 144)
            predictions = model(states_tensor).squeeze(-1) # (B,)
            
            # If the current state itself is solved, target should be 0.0
            states_solved_mask = is_solved_batch(states)
            y[states_solved_mask] = 0.0
            
            loss = loss_fn(predictions, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # 6. Periodically update target network and clear memory cache
            if step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # 7. Periodically evaluate and adjust curriculum
            if step % eval_freq == 0:
                # Evaluate on 100 cubes at current depth with progressive budget for deep scrambles
                budget = 5000 if curriculum_depth == 14 else (3000 if curriculum_depth >= 11 else 1000)
                val_sr = evaluate_model(model, device, curriculum_depth, num_cubes=100, max_nodes=budget)
                
                # Clear memory right after A* evaluation (which creates lots of temp variables)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                elapsed = time.time() - start_time
                print(f"Step {step} | Elapsed: {elapsed:.1f}s | Depth {curriculum_depth} Loss: {loss.item():.4f} | Validation Success Rate: {val_sr:.2f} (max nodes: {budget})")
                
                if val_sr >= success_threshold:
                    print(f"--- Curriculum Level Up! Depth {curriculum_depth} passed with {val_sr*100:.1f}% success rate. ---")
                    curriculum_depth += 1
                    if curriculum_depth > max_depth:
                        print("🎉 CONGRATULATIONS! Model has successfully solved all curriculum depths up to God's Number! 🎉")
                        break
                    print(f"New Scramble Depth: {curriculum_depth}")
                    
                    # Save intermediate weights
                    save_model(model, model_path)
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    
    # Save final model
    save_model(model, model_path)
    print("Training finished.")

if __name__ == '__main__':
    train()
