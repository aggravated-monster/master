from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.wrappers.advise_action import AdviseAction
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.skip_frame import SkipFrame
from mario_phase1.wrappers.track_action import TrackAction
from mario_phase1.wrappers.position_objects import PositionObjects


def apply_wrappers(env, config, detector, positioner, advisor):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"]) # Num of frames to apply one action to
    # The following set of wrappers do not change the observation (it will always be raw pixels)
    # but they use the raw pixel values to perform a series of symbolic transformations on them
    # 3a. Detect objects and store them for later use
    env = DetectObjects(env, detector=detector, seed=config["seed"])  # intercept image and convert to object positions
    # 3b. Translate the bounding boxes to an object/relational representation
    env = PositionObjects(env, positioner=positioner, seed=config["seed"])  # intercept image and convert to object positions
    # 3c. Invoke the Advisor
    env = AdviseAction(env, advisor, seed=config["seed"])
    # 3d. Track the chosen action. This is necessary for the example callbacks
    env = TrackAction(env, seed=config["seed"])
    # From here on, the observation IS altered again, for efficiency purposes in the RL environment
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
    env = GrayScaleObservation(env)
    # 5. Stack frames
    env = FrameStack(env, num_stack=config["stack_size"], lz4_compress=True) # May need to change lz4_compress to False if issues arise

    return env


def apply_wrappers_vanilla(env, config):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"]) # Num of frames to apply one action to
    # 3. Resize dimension
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
    env = GrayScaleObservation(env)
    # 5. Stack frames
    env = FrameStack(env, num_stack=config["stack_size"], lz4_compress=True) # May need to change lz4_compress to False if issues arise

    return env
