# code heavily inspired by Nicholas Renotte's tutorial
# YouTube: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# GitHub: https://github.com/nicknochnack/MarioRL

import gym_super_mario_bros
import numpy as np
import torch
from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace

from mario_stable_baselines_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_stable_baselines_phase1.callbacks.episode_callback import EpisodeCallback
from mario_stable_baselines_phase1.callbacks.interval_callback import IntervalCallback
from mario_stable_baselines_phase1.callbacks.negative_example_callback import NegativeExampleCallback
from mario_stable_baselines_phase1.callbacks.positive_example_callback import PositiveExampleCallback

from our_stable_baselines3 import DQN
from our_stable_baselines3.common.evaluation import evaluate_policy

from mario_stable_baselines_phase1.symbolic_components.advisor import Advisor
from mario_stable_baselines_phase1.symbolic_components.detector import Detector
from mario_stable_baselines_phase1.wrappers.wrappers import apply_wrappers
from mario_stable_baselines_phase1.symbolic_components.positioner import Positioner
# Import PPO for algos
#from our_stable_baselines3 import DQN

from mario_stable_baselines_phase1.our_logging import our_logging

LOG_TIMING = True
our_logging.initialize(LOG_TIMING)

seed = 2

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = True
CHECKPOINT_FREQUENCY = 10000
TOTAL_TIME_STEPS = 10000
CHECKPOINT_DIR = 'train/'
TENSORBOARD_LOG_DIR = 'logs/tensorboard/'

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'


config = {
    "device": device_name,
    "observation_dim": 5*64,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.000001,
    "n_steps": 512,
    "rl_policy": 'MlpPolicy',
    "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../mario_phase0/models/data.yaml',
    "positions_asp": './asp/positions.lp',
    "show_asp": './asp/show.lp',
    "relative_positions_asp": './asp/relative_positions.lp',
    "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
    "generate_examples": True,
    "advice_asp": './asp/advice.lp',
    "show_advice_asp": './asp/show_advice.lp',
}

# Setup game
# 1. Create the object detector. This is a YOLO8 model
detector = Detector(config)
# 2. Create the Translator
positioner = Positioner(config)
# 3. Create the Advisor
advisor = Advisor(config)

# 4. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
# hack the observation space of the environment. We reduce to a single vector, but the environment is expecting
# a colored image. This can be overridden by setting the observation space manually
# We no longer input the bounding boxes, but simply the raw pixel frame
# Maxim might tinker with the inputs, but I don't need to, so I won't.
#env.observation_space = spaces.Box(low=-1, high=1024, shape=(config["observation_dim"],), dtype=np.float32)
print(env.observation_space)

# 5. Apply the decorator chain
env = apply_wrappers(env, config, detector, positioner, advisor)

# 6. Start the game
state = env.reset()

# 7. Setup model saving callback and pass the configuration, so we know the exact configuration belonging to the logs
checkpointCallback = CheckpointCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, config=config)
intervalCallback = IntervalCallback(check_freq=10)
episodeCallback = EpisodeCallback()
negativeExamplesCallback = NegativeExampleCallback()
positiveExamplesCallback = PositiveExampleCallback(check_freq=10)

model = DQN(
    "CnnPolicy",
    env,
    verbose=0,
    train_freq=8,
    gradient_steps=1,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    target_update_interval=10000,
    learning_starts=100,
    buffer_size=100000,
    batch_size=32,
    learning_rate=0.00025,
    policy_kwargs=dict(net_arch=[512, 512]),
    tensorboard_log=TENSORBOARD_LOG_DIR,
    seed=seed,
    device=device,
)

# 8. Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[checkpointCallback,
                                                        intervalCallback,
                                                        episodeCallback,
                                                        negativeExamplesCallback,
                                                        positiveExamplesCallback
                                                        ])

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    deterministic=True,
    n_eval_episodes=20,
)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Training done")

