# code heavily inspired by Nicholas Renotte's tutorial
# YouTube: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# GitHub: https://github.com/nicknochnack/MarioRL

import gym_super_mario_bros
import torch
from nes_py.wrappers import JoypadSpace
from callback import TrainAndLoggingCallback
from wrappers import apply_wrappers
# Import PPO for RL
from stable_baselines3_master.stable_baselines3 import PPO

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = True
CHECKPOINT_FREQUENCY = 10000
TOTAL_TIME_STEPS = 10000
CHECKPOINT_DIR = 'train/'
LOG_DIR = 'logs/'

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)

config = {
    "device": device,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.000001,
    "n_steps": 512,
    "rl_policy": 'CnnPolicy'
}

# Setup game
# 1. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = apply_wrappers(env)

state = env.reset()

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, config=config)

# Initialise the model using PPO as RL algo
model = PPO(config["rl_policy"], env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=config["learning_rate"], n_steps=config["n_steps"])

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callback)

print("Training done")

