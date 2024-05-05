from abc import ABC, abstractmethod

import gym_super_mario_bros
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from nes_py.wrappers import JoypadSpace

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.callbacks.induction_callback import InductionCallback
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.callbacks.negative_example_callback import NegativeExampleCallback
from mario_phase1.callbacks.positive_example_callback import PositiveExampleCallback
from mario_phase1.ddqn.ddqn import DDQN
from mario_phase1.mario_logging import logging
from mario_phase1.symbolic_components.advisor import Advisor
from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.example_collector import NaiveExampleCollector
from mario_phase1.symbolic_components.inducer import Inducer
from mario_phase1.symbolic_components.positioner import Positioner
from mario_phase1.wrappers.advise_action import AdviseAction
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.track_action import TrackAction
from mario_phase1.wrappers.position_objects import PositionObjects


class TestAgent(ABC):

    def __init__(self, config, name=''):
        self.config = config
        # extract the necessary configs for better readability
        self.env_name = config["environment"]
        self.device = config["device"]
        self.check_freq = config["checkpoint_frequency"]
        self.check_dir = config["checkpoint_dir"]
        self.display = config["display"]
        self.name = name

    def __next_model(self, env, seed, advisor=None): # beetje gefrobel met de advisor. Wint geen OO-prijs...
        model = DDQN(env,
                     input_dims=env.observation_space.shape,
                     num_actions=env.num_actions.n,
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
                     device=self.config["device"],
                     advisor=advisor,
                     )

        return model

    @abstractmethod
    def _apply_wrappers(self, env, seed):
        pass

    def _get_callbacks(self) -> list[BaseCallback]:
        return [CheckpointCallback(config=self.config, name=self.name),
                IntervalCallback(check_freq=self.config["interval_frequency"]),
                EpisodeCallback(),
                ]

    def execute(self, num_tests, num_steps, start_seed, advisor=None): # beetje gefrobel met de advisor. Wint geen OO-prijs...
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
            model = self.__next_model(env, seed, advisor)
            # the callbacks are not guaranteed to be stateless either, so get fresh objects for each experiment
            callbacks = self._get_callbacks()
            model.train(min_timesteps_to_train=num_steps, callback=callbacks)


class VanillaAgent(TestAgent):
    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "vanilla")

        super().__init__(config, "vanilla")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env


class DetectionEnabledAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "detection_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        super().__init__(config, "detection_enabled")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
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
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env


class PositionEnabledAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "position_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, "position_enabled")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env


class PositiveExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "positive_examples_producing")

        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        super().__init__(config, "positive_examples_producing")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))

        return callbacks


class NegativeExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "negative_examples_producing")

        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        super().__init__(config, "negative_examples_producing")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))

        return callbacks


class ExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "examples_producing")

        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        super().__init__(config, "examples_producing")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))

        return callbacks


class InductionAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "induction")

        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        self.max_induced_programs = config["max_induced_programs"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        # 3. Create the Inducer
        self.inducer = Inducer(config, bias=config['bias'])
        # 4. Create the Advisor. The Induction Callback needs a reference to the Advisor to force a refresh after induction has finished
        # This is not particularly beautiful
        self.advisor = Advisor(config)
        super().__init__(config, "induction")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use        self.collector = ExampleCollector()
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(InductionCallback(self.inducer, self.advisor,
                                           check_freq=self.symbolic_learn_frequency,
                                           max_induced_programs=self.max_induced_programs))

        return callbacks


class FullyWrappedAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "fully_wrapped")

        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        self.max_induced_programs = config["max_induced_programs"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        # 3. Create the Inducer
        self.inducer = Inducer(config, bias=config['bias'])
        # 4. Create the Advisor. The Induction Callback needs a reference to the Advisor to force a refresh after induction has finished
        # This is not particularly beautiful
        self.advisor = Advisor(config)
        super().__init__(config, "fully_wrapped")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3c. Invoke the Advisor
        env = AdviseAction(env, self.advisor, seed=seed)
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(InductionCallback(self.inducer, self.advisor,
                                           check_freq=self.symbolic_learn_frequency,
                                           max_induced_programs=self.max_induced_programs))

        return callbacks


class FullyIntegratedAgent(TestAgent):

    def __init__(self, config):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        logging.initialize(True, "fully_integrated")

        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        self.max_induced_programs = config["max_induced_programs"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = NaiveExampleCollector()
        # 3. Create the Inducer
        self.inducer = Inducer(config, bias=config['bias'])
        # 4. Create the Advisor. The Induction Callback needs a reference to the Advisor to force a refresh after induction has finished
        # This is not particularly beautiful
        self.advisor = Advisor(config)
        super().__init__(config, "fully_integrated")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # The following set of wrappers do not change the observation (it will always be raw pixels)
        # but they use the raw pixel values to perform a series of symbolic transformations on them
        # 3a. Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed)  # intercept image and convert to object positions
        # 3b. Translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed)  # intercept image and convert to object positions
        # 3d. Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed)
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env)
        # 5. Stack frames
        env = FrameStack(env, num_stack=self.config["stack_size"],
                         lz4_compress=True)  # May need to change lz4_compress to False if issues arise

        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))
        callbacks.append(InductionCallback(self.inducer, self.advisor,
                                           check_freq=self.symbolic_learn_frequency,
                                           max_induced_programs=self.max_induced_programs))

        return callbacks
