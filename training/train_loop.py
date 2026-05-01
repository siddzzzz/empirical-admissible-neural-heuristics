import sys
import os

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.rubiks_env import RubiksEnv
from agents.tactical.ppo_agent import create_tactical_agent, TrajectoryCallback
from agents.tactical.macro_discovery import MacroDiscoverer
from training.curriculum import CurriculumScheduler
from stable_baselines3.common.callbacks import BaseCallback

class CombinedCallback(BaseCallback):
    """
    Handles saving successful trajectories and managing the curriculum difficulty.
    """
    def __init__(self, macro_discoverer, scheduler, verbose=0):
        super(CombinedCallback, self).__init__(verbose)
        self.macro_discoverer = macro_discoverer
        self.scheduler = scheduler
        self.current_trajectory = []
        self.episodes = 0

    def _on_step(self) -> bool:
        # Save action to trajectory
        action = self.locals.get('actions')[0]
        self.current_trajectory.append(action)
        
        # Check if environment is done
        dones = self.locals.get('dones')
        if dones[0]:
            info = self.locals.get('infos')[0]
            is_solved = info.get('is_solved', False)
            
            # Record trajectory if solved
            if is_solved:
                self.macro_discoverer.add_trajectory(self.current_trajectory)
            
            # Record success for curriculum
            self.scheduler.record_result(is_solved)
            
            self.current_trajectory = []
            self.episodes += 1
            
            # Step curriculum periodically
            if self.scheduler.step():
                new_moves = self.scheduler.get_scramble_moves()
                # Update the environment scramble difficulty
                self.training_env.env_method('set_scramble_moves', new_moves)
                
            if self.episodes % 50 == 0:
                sr = self.scheduler.get_success_rate()
                print(f"Episode {self.episodes} | Curriculum Moves: {self.scheduler.get_scramble_moves()} | Success Rate: {sr:.2f}")
                
            if self.episodes % 100 == 0:
                new_macros = self.macro_discoverer.discover_macros()
                if new_macros:
                    print(f"New macros discovered. (Integration into action space pending)")
                    
        return True

def train():
    print("Initializing Rubik's Cube Environment...")
    env = RubiksEnv(scramble_moves=1)
    
    print("Initializing Macro Discoverer...")
    macro_discoverer = MacroDiscoverer()
    
    print("Creating Tactical Agent (PPO)...")
    model = create_tactical_agent(env)
    
    scheduler = CurriculumScheduler(start_moves=1, max_moves=20, threshold_success_rate=0.8, window_size=50)
    callback = CombinedCallback(macro_discoverer, scheduler)

    print("Starting Training Loop (This will actually train the network now)...")
    # Train for a sufficient number of timesteps to see learning on 2x2
    # Since episodes are very short (e.g. 5-20 steps), 100k timesteps is thousands of episodes.
    total_timesteps = 15_000_000 
    model.learn(total_timesteps=total_timesteps, callback=callback)

    print("Training loop complete.")
    
    # Save model
    # Create dir if not exists
    os.makedirs("trained_models", exist_ok=True)
    model_path = "trained_models/tactical_agent"
    print(f"Saving tactical agent to {model_path}...")
    model.save(model_path)

if __name__ == '__main__':
    train()
