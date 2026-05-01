from env.cube import Cube2x2
import numpy as np

def test_moves():
    c = Cube2x2()
    assert c.is_solved(), "Initial state should be solved"
    
    # Test all 18 moves: performing a move 4 times (for 90 deg) or 2 times (for 180 deg) should restore the state
    for i, move in enumerate(c.action_space_names):
        c.reset()
        is_double = '2' in move
        reps = 2 if is_double else 4
        
        for _ in range(reps):
            c.step(i)
            
        if not c.is_solved():
            print(f"Move {move} failed to restore state after {reps} reps!")
            return False

    # Test move and its prime
    for i, move in enumerate(c.action_space_names):
        if "'" in move or '2' in move:
            continue
        
        prime_idx = c.action_space_names.index(move + "'")
        c.reset()
        c.step(i)
        c.step(prime_idx)
        if not c.is_solved():
            print(f"Move {move} followed by {move}' failed to restore state!")
            return False

    # Test random scramble
    c.reset()
    c.scramble(20)
    # Check if colors are preserved (4 of each)
    counts = np.bincount(c.state, minlength=6)
    if not np.all(counts == 4):
        print(f"Scramble altered color counts! Counts: {counts}")
        return False
        
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_moves()
