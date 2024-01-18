import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from callback import TrainAndLoggingCallback
from wrappers import apply_wrappers
# Import PPO for algos
from stable_baselines3 import PPO

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = True
CHECKPOINT_FREQUENCY = 10000
TOTAL_TIME_STEPS = 10000
CHECKPOINT_DIR = 'train/'
LOG_DIR = 'logs/'

# Setup game
# 1. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = apply_wrappers(env)

state = env.reset()

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callback)

print("Training done")

