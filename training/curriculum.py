class CurriculumScheduler:
    """
    Manages the difficulty of the scrambles over the training process.
    """
    def __init__(self, start_moves=1, max_moves=20, threshold_success_rate=0.8, window_size=100):
        self.current_moves = start_moves
        self.max_moves = max_moves
        
        self.threshold = threshold_success_rate
        self.window_size = window_size
        self.history = []

    def record_result(self, is_solved):
        self.history.append(int(is_solved))
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def get_success_rate(self):
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def step(self):
        """
        Called periodically (e.g., end of an episode). 
        Returns True if the difficulty was increased.
        """
        if len(self.history) >= self.window_size:
            if self.get_success_rate() >= self.threshold:
                if self.current_moves < self.max_moves:
                    self.current_moves += 1
                    self.history = [] # Reset history on level up
                    print(f"Curriculum Level Up! Now scaffolding at {self.current_moves} moves.")
                    return True
        return False

    def get_scramble_moves(self):
        return self.current_moves
