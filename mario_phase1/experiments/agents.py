from abc import ABC, abstractmethod

import gym_super_mario_bros
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.callbacks.negative_example_callback import NegativeExampleCallback
from mario_phase1.callbacks.positive_example_callback import PositiveExampleCallback
from mario_phase1.ddqn.ddqn import DDQN
from mario_phase1.mario_logging import logging
from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.positioner import Positioner
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.skip_frame import SkipFrame
from mario_phase1.wrappers.track_action import TrackAction
from mario_phase1.wrappers.translate_objects import PositionObjects


class TestAgent(ABC):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        self.config = config
        self.env_name = env_name
        self.device = device
        self.check_freq = check_freq
        self.check_dir = check_dir
        self.display = display

    def __next_model(self, env, seed):
        model = DDQN(env,
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
                     seed=seed,
                     device=self.device
                     )

        return model

    @abstractmethod
    def _apply_wrappers(self, env, seed):
        pass

    def _get_callbacks(self) -> list[BaseCallback]:
        return [CheckpointCallback(check_freq=self.check_freq, save_path=self.check_dir, config=self.config),
                IntervalCallback(check_freq=1),
                EpisodeCallback(),
                ]

    def execute(self, num_tests, num_steps, start_seed):
        for n in range(num_tests):
            seed = start_seed + n
            # the environment can have a hidden state due to the queues for the symbolic components
            # Hence, start every experiment with its own new environment
            env = gym_super_mario_bros.make(self.env_name, render_mode='human' if self.display else 'rgb',
                                            apply_api_compatibility=True)
            # apply wrappers to track the seed in the timing logging
            env = self._apply_wrappers(env, seed)
            env.reset()
            # To control for possible confounding effects, also supply a new model
            model = self.__next_model(env, seed)
            # the callbacks are not guaranteed to be stateless either, so get fresh objects for each experiment
            callbacks = self._get_callbacks()
            model.train(min_timesteps_to_train=num_steps, callback=callbacks)


class VanillaAgent(TestAgent):
    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "vanilla")

        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        # 1. Simplify the controls
        env = JoypadSpace(env, RIGHT_ONLY)
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env


class DetectionEnabledAgent(TestAgent):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "detecttion_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env


class PositionEnabledAgent(TestAgent):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "position_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        env = TrackAction(env)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env


class PositiveExamplesProducingAgent(TestAgent):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "positive_examples_producing")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        env = TrackAction(env)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(PositiveExampleCallback(check_freq=10, offload_freq=500))

        return callbacks


class NegativeExamplesProducingAgent(TestAgent):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "negative_examples_producing")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        env = TrackAction(env)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(offload_freq=500))

        return callbacks


class ExamplesProducingAgent(TestAgent):

    def __init__(self, config, env_name, device, check_freq, check_dir, display):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, True, "examples_producing")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env_name, device, check_freq, check_dir, display)

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        #env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        env = TrackAction(env)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=False)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(offload_freq=500))
        callbacks.append(PositiveExampleCallback(check_freq=10, offload_freq=500))

        return callbacks
