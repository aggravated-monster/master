import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.experiments.agents_alt import BaselineAgent, NegativeExamplesProducingAgent, PositionEnabledAgent, \
    PositiveExamplesProducingAgent, DetectionEnabledAgent, ExamplesProducingAgent, InductionAgent, FullyIntegratedAgent
from mario_phase1.mario_logging import logging

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
        "skip": 4,
        "stack_size": 4,
        "joypad_space": RIGHT_ONLY,
        "learning_rate": 0.00025,
        "save_replay_buffer": False,
        "detector_model_path": '../../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '../../mario_phase0/models/data.yaml',
        "positions_asp": '../asp/positions.lp',
        "show_asp": '../asp/show.lp',
        "relative_positions_asp": '../asp/relative_positions.lp',
        "show_closest_obstacle_asp": '../asp/show_closest_obstacle.lp',
        "generate_examples": True,
        "advice_asp": '../asp/advice_constraints.lp',
        "show_advice_asp": '../asp/show_advice.lp',
        "ilasp_binary": '../asp/bin/ILASP',
        "ilasp_mode_bias": '../asp/ilasp_mode_bias.las',
        "bias": 'negative',
        "constraints": False,
        "forget": True,
        "positive_examples_frequency": 10,
        "symbolic_learn_frequency": 1000,
        "max_induced_programs": 100
    }


def run(config, num_tests, num_steps, start_seed):

    #NegativeExamplesProducingAgent(config).execute(num_tests, num_steps, start_seed)
    #
    #PositionEnabledAgent(config).execute(num_tests, num_steps, start_seed)
    #
    #PositiveExamplesProducingAgent(config).execute(num_tests, num_steps, start_seed)
    #
    BaselineAgent(config).execute(num_tests, num_steps, start_seed)
    #
    #DetectionEnabledAgent(config).execute(num_tests, num_steps, start_seed)
    #
    #ExamplesProducingAgent(config).execute(num_tests, num_steps, start_seed)
    #
    #InductionAgent(config).execute(num_tests, num_steps, start_seed)
    #
    # Some terribly bad OO practice no one ever needs to see I hope
    # TODO rework
    #fully_integrated_agent = FullyIntegratedAgent(config)
    #fully_integrated_agent.execute(num_tests, num_steps, start_seed, fully_integrated_agent.advisor)


if __name__ == '__main__':
    logging.initialize(True, "experiment")
    run(prepare_config(),
        num_tests=1,
        num_steps=10000,
        start_seed=43)  # the seed is incremented by n*13 in each repetition
    print("Experiment done")
