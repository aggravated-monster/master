# code heavily inspired by Nicholas Renotte's tutorial
# YouTube: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# GitHub: https://github.com/nicknochnack/MarioRL

import gym_super_mario_bros
import numpy as np
import torch
import uuid
from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.evaluation import evaluate_policy

from callback import CheckpointCallback, IntervalCallback, EpisodeCallback
from detector import Detector
from wrappers import apply_wrappers
from positioner import Positioner
# Import PPO for algos
from stable_baselines3 import PPO, DQN

import our_logging

LOG_TIMING = True
our_logging.initialize(LOG_TIMING)

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = True
CHECKPOINT_FREQUENCY = 10000
TOTAL_TIME_STEPS = 10000
CHECKPOINT_DIR = 'train/'
TENSORBOARD_LOG_DIR = 'logs/tensorboard/'

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)

config = {
    "device": device,
    "observation_dim": 5*64,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.000001,
    "n_steps": 512,
    "rl_policy": 'MlpPolicy',
    "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../mario_phase0/models/data.yaml',
    "positions_asp": './asp/positions.lp',
    "show_asp": './asp/show.lp'
}

# Setup game
# 1. Create the object detector. This is a YOLO8 model
detector = Detector(config)
positioner = Positioner(config)

# 2. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
# hack the observation space of the environment. We reduce to a single vector, but the environment is expecting
# a colored image. This can be overridden by setting the observation space manually
env.observation_space = spaces.Box(low=-1, high=1024, shape=(config["observation_dim"],), dtype=np.float32)
print(env.observation_space)

# 3. Apply the decorator chain
env = apply_wrappers(env, config, detector, positioner)

# 4. Start the game
state = env.reset()

# Setup model saving callback and pass the configuration, so we know the exact configuration belonging to the logs
checkpointCallback = CheckpointCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, config=config)
intervalCallback = IntervalCallback(check_freq=10)
episodeCallback = EpisodeCallback()

# This is the AI model started
#model = PPO(config["rl_policy"], env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR, learning_rate=config["learning_rate"], n_steps=config["n_steps"])
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=TENSORBOARD_LOG_DIR,
    seed=2,
)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[checkpointCallback, intervalCallback, episodeCallback])

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20,
)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Training done")

