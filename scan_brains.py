import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.cube import Cube2x2
from agents.tactical.value_agent import ValueNet, load_model, batch_to_tensor

def get_saliency(state: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """
    Computes sticker saliency map by taking the gradient of the predicted distance
    with respect to the input sticker one-hot encodings.
    Returns:
        saliency: numpy array of shape (24,) with values normalized between 0 and 1.
    """
    model.eval()
    
    # Create one-hot representation manually to track gradients
    one_hot = np.zeros((24, 6), dtype=np.float32)
    one_hot[np.arange(24), state] = 1.0
    
    # Create tensor and enable gradient tracking
    tensor_state = torch.tensor(one_hot.flatten(), dtype=torch.float32, requires_grad=True, device=device)
    
    # Forward pass with unsqueeze to add batch dimension
    output = model(tensor_state.unsqueeze(0))
    
    # Backward pass
    model.zero_grad()
    output.backward()
    
    # Extract gradient
    grads = tensor_state.grad.cpu().numpy() # Shape: (144,)
    grads_reshaped = grads.reshape(24, 6)
    
    # Sum absolute gradients across channels to get sticker importance
    saliency = np.sum(np.abs(grads_reshaped), axis=-1)
    
    # Normalize to [0, 1] range
    max_val = np.max(saliency)
    if max_val > 0:
        saliency = saliency / max_val
        
    return saliency

def profile_heuristic(model: torch.nn.Module, device: str = "cpu", num_samples: int = 50, max_depth: int = 14):
    """
    Evaluates heuristic calibration and admissibility at scramble depths from 1 to max_depth.
    """
    cube = Cube2x2()
    # Cache action permutations
    action_perms = [cube._moves[cube.action_space_names[a]] for a in range(18)]
    
    calibration_data = {}
    
    print("\n" + "="*70)
    print("RUNNING NEURAL HEURISTIC CALIBRATION SCAN")
    print("="*70)
    print(f"{'Depth':<6} | {'Mean NN':<10} | {'Std Dev':<10} | {'Min NN':<10} | {'Max NN':<10} | {'Admissible%':<12}")
    print("-"*70)
    
    depths = list(range(1, max_depth + 1))
    mean_estimates = []
    std_estimates = []
    
    for depth in depths:
        estimates = []
        admissible_count = 0
        
        # Scramble and collect predictions
        for _ in range(num_samples):
            state = np.array([0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4, dtype=np.int8)
            for _ in range(depth):
                act = random.randint(0, 17)
                state = state[action_perms[act]]
            
            # Predict value
            with torch.no_grad():
                tensor_state = batch_to_tensor(np.expand_dims(state, 0), device)
                pred = model(tensor_state).item()
                
            estimates.append(max(0.0, pred))
            # If the estimate is <= depth, it satisfies our upper-bound admissibility
            if pred <= depth:
                admissible_count += 1
                
        estimates = np.array(estimates)
        mean_val = np.mean(estimates)
        std_val = np.std(estimates)
        min_val = np.min(estimates)
        max_val = np.max(estimates)
        admissible_rate = (admissible_count / num_samples) * 100.0
        
        mean_estimates.append(mean_val)
        std_estimates.append(std_val)
        
        print(f"{depth:<6} | {mean_val:<10.2f} | {std_val:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f} | {admissible_rate:<11.1f}%")
        
        calibration_data[depth] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "admissible_rate": admissible_rate
        }
    print("="*70)
    
    return depths, mean_estimates, std_estimates, calibration_data

def generate_calibration_plot(depths, mean_estimates, std_estimates, output_path="trained_models/brain_calibration.png"):
    """
    Generates and saves a calibration curve plot comparing True Scramble Depth vs NN Estimates.
    """
    plt.figure(figsize=(8, 5))
    
    # Perfect calibration diagonal
    plt.plot([0] + depths, [0] + depths, 'k--', label="Perfect Heuristic (h(s) = depth)")
    
    # Model predictions
    mean_estimates = [0] + list(mean_estimates)
    std_estimates = [0] + list(std_estimates)
    plot_depths = [0] + list(depths)
    
    plt.errorbar(plot_depths, mean_estimates, yerr=std_estimates, fmt='o-', color='#6366f1',
                 ecolor=(99/255, 102/255, 241/255, 0.3), elinewidth=3, capsize=0, label="ValueNet Heuristic")
                 
    plt.title("Value Network Heuristic Calibration Landscape", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("True Scramble Depth (moves from solved)", fontsize=10)
    plt.ylabel("NN Distance Estimate h(s)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, max(depths) + 1)
    plt.ylim(0, max(depths) + 3)
    plt.legend(loc="upper left")
    
    # Save directory setup
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved successfully to {output_path}")

def run_brain_scan():
    model_path = "trained_models/value_agent.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Train the model first.")
        return
        
    print(f"Loading ValueNet model from {model_path} for brain scan...")
    try:
        model = load_model(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    depths, mean_ests, std_ests, _ = profile_heuristic(model, device=device, num_samples=100)
    generate_calibration_plot(depths, mean_ests, std_ests)
    
    # Saliency Demo
    print("\nDemo: Computing sticker saliency map for a scrambled cube...")
    cube = Cube2x2()
    cube.scramble(10)
    saliency = get_saliency(cube.state, model, device=device)
    
    print("\nNormalized Saliency values per face (4 stickers each):")
    faces = ['U', 'D', 'F', 'B', 'L', 'R']
    for i, face in enumerate(faces):
        face_sal = saliency[i*4:(i+1)*4]
        formatted_sal = ", ".join([f"{v:.2f}" for v in face_sal])
        print(f"  Face {face}: [{formatted_sal}]")
    print("Brain scan finished.")

if __name__ == '__main__':
    run_brain_scan()
