from collections import defaultdict

class MacroDiscoverer:
    """
    Discovers frequently used move sequences (macros) from successful solving trajectories.
    """
    def __init__(self, min_freq=5, max_macro_len=4):
        self.min_freq = min_freq
        self.max_macro_len = max_macro_len
        self.discovered_macros = [] # List of tuples, e.g., (0, 5, 2)
        self.trajectories = [] # List of lists of action indices

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def discover_macros(self):
        """
        Basic Byte-Pair Encoding (BPE) style macro discovery.
        Finds the most frequent adjacent pair of actions in the trajectories.
        """
        if not self.trajectories:
            return []
            
        pair_counts = defaultdict(int)
        
        for traj in self.trajectories:
            if len(traj) < 2:
                continue
            for i in range(len(traj) - 1):
                pair = (traj[i], traj[i+1])
                pair_counts[pair] += 1
                
        if not pair_counts:
            return []
            
        best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])
        
        if best_count >= self.min_freq:
            # We found a new macro!
            if best_pair not in self.discovered_macros:
                self.discovered_macros.append(best_pair)
                print(f"Discovered new macro: {best_pair} (freq: {best_count})")
                
                # Replace the pair in all trajectories with the new macro representation
                # To keep it simple, we don't immediately recursive-replace for n>2 yet,
                # but this is the foundation for dynamic hierarchical action spaces.
                return [best_pair]
        return []

    def get_macros(self):
        return self.discovered_macros
