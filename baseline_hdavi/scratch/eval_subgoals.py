import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rubiks_env import RubiksEnv

env = RubiksEnv(scramble_moves=0)

# 1. Reset to fully solved state
obs, info = env.reset(options={'scramble_moves': 0})
print("Initial Subgoals:", env.achieved_subgoals)

# 2. Scramble by 1 move (e.g., U) to break full solve but maintain OLL (partially)
# Wait, 'U' breaks the sides of the first layer, but not OLL (bottom is still yellow).
env.step(0) # 'U'
print("After 'U' - Subgoals:", env.achieved_subgoals)

# 3. Fix it with U'
obs, reward, terminated, truncated, info = env.step(1) # 'U''
print("After 'U'' - Reward:", reward)
print("After 'U'' - Terminated:", terminated)
print("After 'U'' - Subgoals:", env.achieved_subgoals)
