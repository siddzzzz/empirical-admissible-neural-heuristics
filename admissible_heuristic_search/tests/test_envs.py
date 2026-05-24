import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admissible_heuristic_search.envs.cube2x2 import Cube2x2
from admissible_heuristic_search.envs.tile_puzzle import TilePuzzle
from admissible_heuristic_search.envs.lights_out import LightsOut

def test_cube():
    print("Testing Cube2x2...")
    cube = Cube2x2(use_macros=False)
    assert cube.is_solved(), "Cube should start solved!"
    
    # Scramble
    cube.scramble(5)
    assert not cube.is_solved(), "Cube should not be solved after scramble!"
    
    # Reset
    cube.reset()
    assert cube.is_solved(), "Cube should be solved after reset!"
    
    # One-hot encoding shape checks
    state = cube.get_state()
    oh_single = cube.to_one_hot(state)
    assert oh_single.shape == (144,), f"Expected (144,), got {oh_single.shape}"
    
    batch = np.array([state, state])
    oh_batch = cube.to_one_hot(batch)
    assert oh_batch.shape == (2, 144), f"Expected (2, 144), got {oh_batch.shape}"
    
    assert cube.get_base_heuristic(state) == 0.0, "Expected base heuristic 0.0 for Cube."
    print("  Cube2x2 tests passed!")

def test_tile_puzzle():
    print("Testing TilePuzzle...")
    puzzle = TilePuzzle(N=3)
    assert puzzle.is_solved(), "Puzzle should start solved!"
    
    # Scramble
    puzzle.scramble(10)
    assert not puzzle.is_solved(), "Puzzle should not be solved after scramble!"
    
    # Reset
    puzzle.reset()
    assert puzzle.is_solved(), "Puzzle should be solved after reset!"
    
    # One-hot encoding shape checks
    state = puzzle.get_state()
    oh_single = puzzle.to_one_hot(state)
    assert oh_single.shape == (81,), f"Expected (81,), got {oh_single.shape}"
    
    batch = np.array([state, state])
    oh_batch = puzzle.to_one_hot(batch)
    assert oh_batch.shape == (2, 81), f"Expected (2, 81), got {oh_batch.shape}"
    
    # Test Manhattan Distance
    puzzle.reset()
    # Initial solved state should have Manhattan distance 0
    assert puzzle.get_base_heuristic(puzzle.get_state()) == 0.0, "Expected Manhattan distance 0 for solved state."
    
    # Swap blank (idx 8) with tile above it (idx 5, which is 6)
    puzzle.step(0) # UP
    dist = puzzle.get_base_heuristic(puzzle.get_state())
    assert dist == 1.0, f"Expected Manhattan distance 1.0, got {dist}"
    
    print("  TilePuzzle tests passed!")

def test_lights_out():
    print("Testing LightsOut...")
    puzzle = LightsOut(W=3, H=3)
    assert puzzle.is_solved(), "LightsOut should start solved (all off)!"
    
    # Scramble
    puzzle.scramble(3)
    assert not puzzle.is_solved(), "LightsOut should not be solved after scramble!"
    
    # Reset
    puzzle.reset()
    assert puzzle.is_solved(), "LightsOut should be solved after reset!"
    
    # One-hot encoding shape checks
    state = puzzle.get_state()
    oh_single = puzzle.to_one_hot(state)
    assert oh_single.shape == (18,), f"Expected (18,), got {oh_single.shape}"
    
    batch = np.array([state, state])
    oh_batch = puzzle.to_one_hot(batch)
    assert oh_batch.shape == (2, 18), f"Expected (2, 18), got {oh_batch.shape}"
    
    # Test base heuristic
    puzzle.reset()
    assert puzzle.get_base_heuristic(puzzle.get_state()) == 0.0, "Expected base heuristic 0.0 for solved state."
    
    # Toggle middle light (cell 4) - toggles 4 and neighbors (1, 3, 5, 7) -> 5 lights on
    puzzle.step(4)
    h_val = puzzle.get_base_heuristic(puzzle.get_state())
    assert h_val == 1.0, f"Expected base heuristic 1.0 for 5 active lights, got {h_val}"
    
    print("  LightsOut tests passed!")

if __name__ == '__main__':
    test_cube()
    test_tile_puzzle()
    test_lights_out()
    print("All environment tests passed successfully!")
