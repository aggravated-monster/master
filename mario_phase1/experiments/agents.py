from codetiming import Timer
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.ddqn.ddqn import DDQN
from mario_phase1.mario_logging import logging
from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.positioner import Positioner
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.skip_frame import SkipFrame
from mario_phase1.wrappers.translate_objects import PositionObjects
from abc import ABC, abstractmethod


class TestAgent(ABC):

    def __init__(self, config, env, device, check_freq, check_dir):

        self.config = config
        self.env = env
        self._apply_wrappers(env)
        self.device = device
        self.checkpointCallback = CheckpointCallback(check_freq=check_freq, save_path=check_dir,
                                                        config=config)
        self.intervalCallback = IntervalCallback(check_freq=1)
        self.episodeCallback = EpisodeCallback()

    def __next_model(self, seed):
        model = DDQN(self.env,
                     input_dims=self.env.observation_space.shape,
                     num_actions=self.env.action_space.n,
                     lr=0.00025,
                     gamma=0.9,
                     epsilon=1.0,
                     eps_decay=0.99999975,
                     eps_min=0.1,
                     replay_buffer_capacity=50000,
                     batch_size=32,
                     sync_network_rate=10000,
                     verbose=1,
                     seed=seed,
                     device=self.device
                     )

        return model

    @abstractmethod
    def _apply_wrappers(self, env):
        pass

    def _get_callbacks(self):
        return [self.checkpointCallback,
                self.intervalCallback,
                self.episodeCallback,
                ]

    def execute(self, num_tests, num_steps, start_seed=2):
        for n in range(num_tests):
            # TODO possibly have to recreate environment altogether because of hidden state introduced by symbolic components
            self.env.reset()
            model = self.__next_model(start_seed+n)
            callbacks = self._get_callbacks()
            model.train(min_timesteps_to_train=num_steps, callback=callbacks)


class VanillaAgent(TestAgent):
    def __init__(self, config, env, device, check_freq, check_dir):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True,"vanilla")

        super().__init__(config, env, device, check_freq, check_dir)

    def _apply_wrappers(self, env):
        # 1. Simplify the controls
        env = JoypadSpace(env, RIGHT_ONLY)
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        self.env = env


class PositionEnabledAgent(TestAgent):

    def __init__(self, config, env, device, check_freq, check_dir):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "position_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env, device, check_freq, check_dir)

    def _apply_wrappers(self, env):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner)  # intercept image and convert to object positions
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        self.env = env
