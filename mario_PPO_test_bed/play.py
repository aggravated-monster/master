import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
# Import PPO for algos
from stable_baselines3_master.stable_baselines3 import PPO

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'

# Setup game
# 1. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = apply_wrappers(env)


# Load model
model = PPO.load('./train/best_model_10000')
# Start the game
state = env.reset()
# Loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
