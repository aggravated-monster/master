import json

import gym_super_mario_bros
import numpy as np
import torch
from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace

from mario_stable_baselines_phase1.symbolic_components.detector import Detector
from mario_stable_baselines_phase1.wrappers.wrappers import apply_wrappers
# Import PPO for algos
from master_stable_baselines3 import PPO, DQN

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)

# load the configuration belonging to the model we want to use
with open('./train/configuration.json', 'r') as file:
    config = json.load(file)


# Setup game
# 1. Create the object detector. This is a YOLO8 model
#detector = Detector(config)

# 2. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
# hack the observation space of the environment. We reduce to a single vector, but the environment is expecting
# a colored image. This can be overridden by setting the observation space manually
#env.observation_space = spaces.Box(low=-1, high=1024, shape=(config["observation_dim"],), dtype=np.float32)
#print(env.observation_space)

# 3. Apply the decorator chain
env = apply_wrappers(env, config, None, None, None)

# 4. Load model
model = DQN.load('./train/best_model_1000000')

# 5. Start the game
state = env.reset()

# Loop through the game until we kill it
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
10000