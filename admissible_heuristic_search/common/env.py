import abc
import numpy as np

class CombinatorialPuzzle(abc.ABC):
    """
    Abstract Base Class for combinatorial puzzles.
    Every puzzle must implement this interface to work with the generalized A* solver and training loop.
    """
    
    @abc.abstractmethod
    def reset(self):
        """Resets the puzzle to the canonical solved state and returns the state."""
        pass
        
    @abc.abstractmethod
    def scramble(self, depth: int) -> np.ndarray:
        """Scrambles the puzzle from the solved state using `depth` random actions."""
        pass
        
    @abc.abstractmethod
    def step(self, action_idx: int) -> float:
        """
        Applies action at index `action_idx` to the current state.
        Returns the transition cost (g-cost).
        """
        pass
        
    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """Returns the current state representation (numpy array)."""
        pass
        
    @abc.abstractmethod
    def set_state(self, state: np.ndarray):
        """Sets the puzzle state to a specific configuration."""
        pass
        
    @abc.abstractmethod
    def is_solved(self) -> bool:
        """Returns True if the current state is solved, False otherwise."""
        pass
        
    @property
    @abc.abstractmethod
    def action_space_names(self) -> list:
        """Returns a list of action name strings."""
        pass
        
    @property
    def num_actions(self) -> int:
        return len(self.action_space_names)
        
    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Returns the flat dimensionality of the raw state."""
        pass
        
    @property
    @abc.abstractmethod
    def one_hot_dim(self) -> int:
        """Returns the flat dimensionality of the one-hot encoded state."""
        pass
        
    @abc.abstractmethod
    def to_one_hot(self, state: np.ndarray) -> np.ndarray:
        """Converts raw state representation to one-hot representation."""
        pass
        
    @abc.abstractmethod
    def get_base_heuristic(self, state: np.ndarray) -> float:
        """
        Returns a simple analytical admissible base heuristic h0(s).
        Must be strictly admissible (h0(s) <= h*(s)).
        If no analytical heuristic is available, return 0.0.
        """
        pass
