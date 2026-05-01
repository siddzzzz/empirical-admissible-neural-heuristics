import numpy as np

def calculate_entropy(state):
    """
    Calculate the disorder (entropy) of the cube state.
    Lower entropy means the cube is closer to being solved.
    For a 2x2 cube without centers, we calculate the disorder of each face
    by finding the most frequent color on that face and counting how many 
    stickers do NOT match it.
    
    Max possible entropy for 2x2: 6 faces * 3 mismatches = 18
    Min possible entropy: 0 (solved)
    """
    entropy = 0
    for i in range(6):
        face_stickers = state[i*4:(i+1)*4]
        counts = np.bincount(face_stickers, minlength=6)
        majority_count = counts.max()
        entropy += (4 - majority_count)
    return entropy

def compute_reward(old_state, new_state, is_solved, move_cost=1):
    """
    Compute reward based on solving and entropy reduction.
    """
    if is_solved:
        return 100.0
        
    old_entropy = calculate_entropy(old_state)
    new_entropy = calculate_entropy(new_state)
    
    # Entropy reduction reward (small shaping reward)
    entropy_diff = old_entropy - new_entropy
    
    # Step penalty to encourage shorter solutions.
    # We scale it by move_cost so the agent doesn't spam macros pointlessly.
    step_penalty = -0.1 * move_cost
    
    return (entropy_diff * 1.0) + step_penalty
