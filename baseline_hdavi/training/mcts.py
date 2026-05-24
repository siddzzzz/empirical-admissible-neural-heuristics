import math
import numpy as np

class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_probs, env_step_fn):
        for action, prob in action_probs.items():
            if action not in self.children:
                # To really do this we need a copy of the environment or deterministic transitions
                next_state = env_step_fn(self.state, action)
                self.children[action] = Node(next_state, parent=self, action_taken=action, prior=prob)

class MCTS:
    """
    Monte Carlo Tree Search for Rubik's Cube.
    Used for planning high-level sequences (search-augmented reasoning).
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct

    def search(self, root_state, policy_fn, value_fn, env_step_fn, num_simulations=50):
        root = Node(root_state)
        
        for _ in range(num_simulations):
            node = root
            # Selection
            while node.is_expanded():
                node = self.select_child(node)
                
            # Expansion & Evaluation
            action_probs = policy_fn(node.state)
            value = value_fn(node.state)
            
            node.expand(action_probs, env_step_fn)
            
            # Backpropagation
            self.backpropagate(node, value)
            
        # Return action with highest visit count
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_child

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
