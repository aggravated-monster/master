import collections
import os
import pickle

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
from gym.wrappers import FrameStack, ResizeObservation
from gym_super_mario_bros.actions import RIGHT_ONLY
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm

from mario_phase1.callbacks.checkpoint_callback_alt import CheckpointCallbackAlt
from mario_phase1.callbacks.episode_callback_alt import EpisodeCallbackAlt
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from mario_phase1.mario_logging import logging
from mario_phase1.wrappers.wrappers import apply_wrappers_baseline

#JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize(name="baseline_0_99997_decay")


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'models_baseline/',
        "display": False,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
        "save_replay_buffer": False
    }


def run(config, total_time_steps):
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)
    # Load level
    env = apply_wrappers_baseline(env, config)  # Wraps the environment so that frames are grayscale / segmented
    env.reset()

    checkpoint_callback = CheckpointCallbackAlt(config)
    interval_callback = IntervalCallback(config["interval_frequency"])
    episode_callback = EpisodeCallbackAlt()

    agent = DQNAgent(env,
                     input_dims=env.observation_space.shape,
                     num_actions=env.action_space.n,
                     max_memory_size=4000,
                     batch_size=16,
                     gamma=0.90,
                     lr=config["learning_rate"],
                     dropout=0.,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99997,
                     pretrained=False,
                     verbose=1,
                     seed=config["seed"]
                     )

    agent.train_episodes(num_episodes=5000, callback=[checkpoint_callback,
                                                      interval_callback,
                                                      episode_callback,
                                                      ])

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=1), total_time_steps=1000000)
    #run(prepare_config(seed=13), total_time_steps=1000000)
    #run(prepare_config(seed=42), total_time_steps=1000000)
    #run(prepare_config(seed=21), total_time_steps=1000000)
    #run(prepare_config(seed=2047), total_time_steps=1000000)
    print("Training done")
