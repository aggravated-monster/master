import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from mario_phase1.experiments.agents import PositionEnabledAgent, VanillaAgent, DetectionEnabledAgent, \
    ExamplesProducingAgent, PositiveExamplesProducingAgent, NegativeExamplesProducingAgent, InductionAgent, \
    FullyIntegratedAgent

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'

print(device_name)

config = {
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
    "positive_examples_frequency": 10,
    "symbolic_learn_frequency": 1000,
    "max_induced_programs": 100
}

NUM_TESTS = 2
NUM_STEPS = 1500
START_SEED = 42  # the seed is incremented by 1 in each repetition

def run():
    # NegativeExamplesProducingAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # PositionEnabledAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # PositiveExamplesProducingAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # VanillaAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # DetectionEnabledAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # ExamplesProducingAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    # InductionAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)
    #
    FullyIntegratedAgent(config).execute(NUM_TESTS, NUM_STEPS, START_SEED)

if __name__ == '__main__':
    run()
    print("Experiment done")
