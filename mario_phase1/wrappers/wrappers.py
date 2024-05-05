from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.wrappers.advise_action import AdviseAction
from mario_phase1.wrappers.buffer_wrapper import BufferWrapper
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.image_to_pytorch import ImageToPyTorch
from mario_phase1.wrappers.max_and_skip import MaxAndSkipEnv
from mario_phase1.wrappers.resize_and_grayscale import ResizeAndGrayscale
from mario_phase1.wrappers.scaled_float_frame import ScaledFloatFrame
from mario_phase1.wrappers.track_action import TrackAction
from mario_phase1.wrappers.position_objects import PositionObjects


def apply_wrappers_baseline(env, config):
    env = JoypadSpace(env, RIGHT_ONLY)
    env = MaxAndSkipEnv(env)
    env = ResizeAndGrayscale(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 6)
    env = ScaledFloatFrame(env)

    return env

def apply_wrappers(env, config, detector, positioner, advisor):
    env = JoypadSpace(env, RIGHT_ONLY)
    env = MaxAndSkipEnv(env)
    # 3a. Detect objects and store them for later use
    env = DetectObjects(env, detector=detector, seed=config["seed"])  # intercept image and convert to object positions
    # 3b. Translate the bounding boxes to an object/relational representation
    env = PositionObjects(env, positioner=positioner, seed=config["seed"])  # intercept image and convert to object positions
    # 3c. Invoke the Advisor
    #env = AdviseAction(env, advisor, seed=config["seed"])
    # 3d. Track the chosen action. This is necessary for the example callbacks
    env = TrackAction(env, seed=config["seed"])
    env = ResizeAndGrayscale(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 6)
    env = ScaledFloatFrame(env)

    return env
