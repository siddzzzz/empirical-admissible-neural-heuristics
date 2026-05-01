import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rubiks_env import RubiksEnv
from env.cube import Cube2x2

env = RubiksEnv(scramble_moves=0)
cube = Cube2x2()

success = True
for i, move in enumerate(cube.action_space_names):
    # Determine inverse move
    if "'" in move:
        inv_move = move.replace("'", "")
    elif "2" in move:
        inv_move = move
    else:
        inv_move = move + "'"
    
    inv_idx = cube.action_space_names.index(inv_move)
    
    # Reset and apply move
    obs, info = env.reset(options={'scramble_moves': 0})
    env.step(i) # Apply move i
    obs, reward, done, trunc, info = env.step(inv_idx) # Apply inverse
    
    if not info.get('is_solved'):
        print(f"FAILED: Move {move} (idx {i}) followed by {inv_move} (idx {inv_idx}) did not solve!")
        success = False

if success:
    print("Environment logic verified: all 1-move scrambles are perfectly solvable by inverse actions.")
