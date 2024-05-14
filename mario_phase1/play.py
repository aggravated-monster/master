import os

import gym_super_mario_bros
import torch

from mario_logging import logging
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from mario_phase1.ddqn.ddqn_constraints import DQNAgentConstraints
from mario_phase1.symbolic_components.advisor import Advisor
from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.example_collector import ConstraintsExampleCollector, NaiveExampleCollector
from mario_phase1.symbolic_components.positioner import Positioner
from wrappers.wrappers import apply_wrappers_baseline, apply_wrappers, apply_wrappers

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize(name="play-symbolic_0_999985_constraints_target_eps")


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
        "display": False,
        "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '../mario_phase0/models/data.yaml',
        "positions_asp": './asp/positions.lp',
        "show_asp": './asp/show.lp',
        "relative_positions_asp": './asp/relative_positions.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
        "generate_examples": True,
        "advice_asp": './asp/advice_constraints.lp',
        "show_advice_asp": './asp/show_advice.lp',
        "constraints": True
    }


def run(config, num_episodes):
    # Setup game
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    # Setup game
    detector = Detector(config)
    positioner = Positioner(config)

    env = apply_wrappers(env, config, detector, positioner, None)
    env.reset()

    episode_callback = EpisodeCallback()

    agent = DQNAgentConstraints(env,
                                input_dims=env.observation_space.shape,
                                num_actions=env.action_space.n,
                                max_memory_size=4000,
                                batch_size=16,
                                gamma=0.90,
                                lr=config["learning_rate"],
                                dropout=0.,
                                exploration_max=1.0,
                                exploration_min=0.02,
                                exploration_decay=0.999985,
                                pretrained=False,
                                verbose=0,
                                seed=config["seed"],
                                advisor=None
                                )

    agent.load_model(path=os.path.join("./models", "20240421-16.01.26_baseline_fast_decay_1000000_1_target_net.pt"))
    agent.epsilon = 0.02
    agent.exploration_min = 0.0
    agent.exploration_decay = 0.0

    agent.play(num_episodes, callback=episode_callback)

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=1),
        num_episodes=100)
    print("Game done")
