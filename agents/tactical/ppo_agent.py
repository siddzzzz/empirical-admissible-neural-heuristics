from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from agents.tactical.macro_discovery import MacroDiscoverer

class TrajectoryCallback(BaseCallback):
    """
    Callback to save successful trajectories for Macro Discovery.
    """
    def __init__(self, macro_discoverer, verbose=0):
        super(TrajectoryCallback, self).__init__(verbose)
        self.macro_discoverer = macro_discoverer
        self.current_trajectory = []

    def _on_step(self) -> bool:
        # Action is in locals
        action = self.locals.get('actions')[0]
        self.current_trajectory.append(action)
        
        # Check if environment is done
        dones = self.locals.get('dones')
        if dones[0]:
            info = self.locals.get('infos')[0]
            if info.get('is_solved', False):
                # Successful trajectory!
                self.macro_discoverer.add_trajectory(self.current_trajectory)
            self.current_trajectory = []
        return True

def create_tactical_agent(env, device="auto"):
    """
    Creates a PPO agent for the Rubik's Cube environment.
    """
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, # Encourage exploration
        device=device
    )
    return model
