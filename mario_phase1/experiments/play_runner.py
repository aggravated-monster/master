import os

import gym_super_mario_bros
import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from mario_phase1.ddqn.ddqn_constraints import DQNAgentConstraints
from mario_phase1.experiments.agents import PositionEnabledAgent, VanillaAgent, DetectionEnabledAgent, \
    ExamplesProducingAgent, PositiveExamplesProducingAgent, NegativeExamplesProducingAgent, InductionAgent, \
    FullyIntegratedAgent, FullyWrappedAgent
from mario_phase1.mario_logging import logging
from mario_phase1.symbolic_components.advisor import Advisor
import glob

from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.positioner import Positioner
from mario_phase1.wrappers.wrappers import apply_wrappers, apply_wrappers_baseline

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'

print(device_name)


def prepare_config():
    return {
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'test/',
        "display": False,
        "skip": 1,
        "stack_size": 4,
        "joypad_space": RIGHT_ONLY,
        "learning_rate": 0.00025,
        "detector_model_path": '../../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '../../mario_phase0/models/data.yaml',
        "positions_asp": '../asp/positions.lp',
        "show_asp": '../asp/show.lp',
        "relative_positions_asp": '../asp/relative_positions.lp',
        "show_closest_obstacle_asp": '../asp/show_closest_obstacle.lp',
        "generate_examples": True,
        "show_advice_asp": '../asp/show_advice.lp',
        "ilasp_binary": '../asp/bin/ILASP',
        "ilasp_mode_bias": '../asp/ilasp_mode_bias.las',
        "bias": 'positive',
        "positive_examples_frequency": 1,
        "symbolic_learn_frequency": 10000,
        "max_induced_programs": 100
    }


def play_baseline(config, model_folder, num_tests, num_episodes, start_seed):
    # list all target models in directory
    models = glob.glob(model_folder + '*target*')
    # Setup game
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    # Setup game
    env = apply_wrappers_baseline(env, config)
    print(models)
    for model_path in models:

        model_file = os.path.basename(model_path)
        model_name = os.path.splitext(model_file)
        parts = os.path.normpath(model_folder).split(os.path.sep)

        logging.initialize(True, 'comp_' + model_name[0])

        for n in range(num_tests):
            episode_callback = EpisodeCallback()
            seed = start_seed + (13 * n)
            # load the model (with new seed each time, so reload)
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
                             exploration_decay=0.999961,
                             pretrained=False,
                             verbose=0,
                             seed=seed,
                             advisor=None,
                             name=model_name[0]
                             )

            agent.load_model(path=model_path)
            agent.epsilon = 0.02
            agent.exploration_min = 0.0
            agent.exploration_decay = 0.0

            env.reset()

            agent.play(num_episodes, callback=episode_callback)

    env.close()


if __name__ == '__main__':
    print(os.getcwd())
    play_baseline(prepare_config(),
                  model_folder="../../results/last_models/",
                  num_tests=50,
                  num_episodes=10,
                  start_seed=2)
    print("Experiment done")
