import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.experiments.agents import PositionerAgent, BaselineAgent, DetectorAgent, CollectorAgent, \
    InductionAgent, TargetAgent, FullyIntegratedAgent
from mario_phase1.mario_logging import logging
from mario_phase1.mario_logging.logging import Logging

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
        "checkpoint_frequency": 10000000,
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
        "relative_positions_asp": '../asp/relative_positions_ext.lp',
        "show_closest_obstacle_asp": '../asp/show_closest_obstacle_ext.lp',
        "generate_examples": True,
        "advice_asp": '../asp/advise_ext.lp',
        "show_advice_asp": '../asp/show_advice.lp',
        "ilasp_binary": '../asp/bin/ILASP',
        "ilasp_mode_bias": '../asp/ilasp_mode_bias_compact_ext.las',
        "bias": 'positive',
        "constraints": False,
        "forget": True,
        "positive_examples_frequency": 10,
        "symbolic_learn_frequency": 1000,
        "max_induced_programs": 10000
    }


#"advice_asp": '../asp/advice_ext.lp',

def run(config, num_tests, num_steps, start_seed):

        #BaselineAgent(config).execute(num_tests, num_steps, start_seed)
        #
        #PositionerAgent(config).execute(num_tests, num_steps, start_seed)
        #

        #CollectorAgent(config).execute(num_tests, num_steps, start_seed)

        #DetectorAgent(config).execute(num_tests, num_steps, start_seed)
        #

        target_agent = TargetAgent(config)
        target_agent.execute(num_tests, num_steps, start_seed, target_agent.advisor)

        # Some terribly bad OO practice no one ever needs to see I hope
        #fully_integrated_agent = FullyIntegratedAgent(config)
        #fully_integrated_agent.execute(num_tests, num_steps, start_seed, fully_integrated_agent.advisor)

        #InductionAgent(config).execute(num_tests, num_steps, start_seed)





if __name__ == '__main__':
    logging.initialize(True, "timing", True)
    run(prepare_config(),
        num_tests=5,
        num_steps=10000,
        start_seed=7)
    print("Experiment done")
