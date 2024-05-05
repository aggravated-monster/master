import gym_super_mario_bros
import torch

from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
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

logging.initialize(name="baseline_B2")


def prepare_config(seed=1):
    return {
        "name": "baseline_B2",
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'models_baseline/',
        "display": False,
        "learning_rate": 0.00025,
        "save_replay_buffer": False
    }


def run(config, total_episodes):
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)
    # Load level
    env = apply_wrappers_baseline(env, config)  # Wraps the environment so that frames are grayscale / segmented
    env.reset()

    checkpoint_callback = CheckpointCallback(config)
    interval_callback = IntervalCallback(config["interval_frequency"])
    episode_callback = EpisodeCallback()

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
                     exploration_decay=0.9999961,
                     #exploration_decay=0.99,
                     pretrained=False,
                     verbose=0,
                     seed=config["seed"],
                     name=config["name"]
                     )

    agent.train_episodes(num_episodes=total_episodes, callback=[checkpoint_callback,
                                                                interval_callback,
                                                                episode_callback,
                                                                ])

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=13), total_episodes=5000)
    print("Training done")
