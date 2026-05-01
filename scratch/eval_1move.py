import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from env.rubiks_env import RubiksEnv

env = RubiksEnv(scramble_moves=1)
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50000)

successes = 0
for i in range(100):
    obs, info = env.reset(options={'scramble_moves': 1})
    for step in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        if done:
            if info.get('is_solved'):
                successes += 1
            break
print(f"Success rate on 1-move scrambles after 50k steps: {successes}%")
