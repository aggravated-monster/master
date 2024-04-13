import gym_super_mario_bros
import torch

from mario_logging import logging
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.episode_callback import EpisodeCallback
from callbacks.interval_callback import IntervalCallback
from ddqn.ddqn_vanilla import DDQNVanilla
from wrappers.wrappers import apply_wrappers_vanilla

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize(name="vanilla")


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'models_vanilla/',
        "display": False,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
    }


def run(config, total_time_steps):
    # Setup game
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    env = apply_wrappers_vanilla(env, config)
    env.reset()

    checkpoint_callback = CheckpointCallback(config)
    interval_callback = IntervalCallback(config["interval_frequency"])
    episode_callback = EpisodeCallback()

    agent = DDQNVanilla(env,
                        input_dims=env.observation_space.shape,
                        num_actions=env.action_space.n,
                        lr=0.00025,
                        gamma=0.9,
                        epsilon=1.0,
                        eps_decay=0.99999975,
                        eps_min=0.1,
                        replay_buffer_capacity=50000,
                        batch_size=32,
                        sync_network_rate=10000,
                        verbose=1,
                        seed=config["seed"],
                        device=device)

    agent.train(min_timesteps_to_train=total_time_steps, callback=[checkpoint_callback,
                                                                   interval_callback,
                                                                   episode_callback,
                                                                   ])

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=51),
        total_time_steps=10000)
    print("Training done")
