import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.cube import Cube2x2

def test_macro(macro_name, move_string):
    c1 = Cube2x2()
    c2 = Cube2x2()
    
    # 1. Apply macro as single action
    macro_idx = c1.action_space_names.index(macro_name)
    c1.step(macro_idx)
    
    # 2. Apply primitive moves sequentially
    moves = move_string.split()
    for move in moves:
        move_idx = c2.action_space_names.index(move)
        c2.step(move_idx)
        
    if np.array_equal(c1.state, c2.state):
        print(f"PASS: Macro {macro_name} is mathematically identical to sequential execution.")
    else:
        print(f"FAIL: Macro {macro_name} state mismatch!")

test_macro("Sexy", "R U R' U'")
test_macro("Sune", "R U R' U R U2 R'")
test_macro("J-Perm", "R U R' F' R U R' U' R' F R2 U' R' U'")
test_macro("Y-Perm", "F R U' R' U' R U R' F' R U R' U' R' F R F'")
