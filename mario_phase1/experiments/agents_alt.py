from abc import ABC, abstractmethod

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.callbacks.induction_callback import InductionCallback
from mario_phase1.callbacks.interval_callback import IntervalCallback
from mario_phase1.callbacks.negative_example_callback import NegativeExampleCallback
from mario_phase1.callbacks.positive_example_callback import PositiveExampleCallback
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from mario_phase1.mario_logging import logging
from mario_phase1.symbolic_components.advisor import Advisor
from mario_phase1.symbolic_components.detector import Detector
from mario_phase1.symbolic_components.example_collector import ExampleCollector
from mario_phase1.symbolic_components.inducer import Inducer
from mario_phase1.symbolic_components.positioner import Positioner
from mario_phase1.wrappers.buffer_wrapper import BufferWrapper
from mario_phase1.wrappers.detect_objects import DetectObjects
from mario_phase1.wrappers.image_to_pytorch import ImageToPyTorch
from mario_phase1.wrappers.max_and_skip import MaxAndSkipEnv
from mario_phase1.wrappers.position_objects import PositionObjects
from mario_phase1.wrappers.resize_and_grayscale import ResizeAndGrayscale
from mario_phase1.wrappers.scaled_float_frame import ScaledFloatFrame
from mario_phase1.wrappers.track_action import TrackAction


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
        model = DQNAgent(env,
                         input_dims=env.observation_space.shape,
                         num_actions=env.action_space.n,
                         max_memory_size=4000,
                         batch_size=16,
                         gamma=0.90,
                         lr=self.config["learning_rate"],
                         dropout=0.,
                         exploration_max=1.0,
                         exploration_min=0.02,
                         exploration_decay=0.99997,
                         pretrained=False,
                         verbose=1,
                         seed=seed,
                         advisor=advisor,
                         name=self.name
                         )

        return model

    @abstractmethod
    def _apply_wrappers(self, env, seed):
        pass

    def _get_callbacks(self) -> list[BaseCallback]:
        return [CheckpointCallback(config=self.config, name=self.name),
                #IntervalCallback(check_freq=self.config["interval_frequency"]),
                EpisodeCallback(),
                ]

    def execute(self, num_tests, num_steps, start_seed, advisor=None): # beetje gefrobel met de advisor. Wint geen OO-prijs...
        for n in range(num_tests):
            seed = start_seed + (13 * n)
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


class BaselineAgent(TestAgent):
    def __init__(self, config):
        super().__init__(config, "baseline")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
        return env


class DetectionEnabledAgent(TestAgent):

    def __init__(self, config):
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        super().__init__(config, "detector")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        # Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
        return env


class PositionEnabledAgent(TestAgent):

    def __init__(self, config):
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, "positioner")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        # Detect objects and store them for later use
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
        return env


class PositiveExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = ExampleCollector()
        super().__init__(config, "collector positive")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(PositiveExampleCallback(self.collector,
                                                 check_freq=self.positive_examples_frequency,
                                                 offload_freq=self.symbolic_learn_frequency))

        return callbacks


class NegativeExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = ExampleCollector()
        super().__init__(config, "collector negqtive")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
        return env

    def _get_callbacks(self):
        callbacks = super()._get_callbacks()

        callbacks.append(NegativeExampleCallback(self.collector,
                                                 offload_freq=self.symbolic_learn_frequency))

        return callbacks


class ExamplesProducingAgent(TestAgent):

    def __init__(self, config):
        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = ExampleCollector()
        super().__init__(config, "collector")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
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
        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        self.max_induced_programs = config["max_induced_programs"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = ExampleCollector()
        # 3. Create the Inducer
        self.inducer = Inducer(config)
        # 4. Create the Advisor. The Induction Callback needs a reference to the Advisor to force a refresh after induction has finished
        # This is not particularly beautiful
        self.advisor = Advisor(config)
        super().__init__(config, "inducer")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
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
        self.positive_examples_frequency = config["positive_examples_frequency"]
        self.symbolic_learn_frequency = config["symbolic_learn_frequency"]
        self.max_induced_programs = config["max_induced_programs"]
        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        self.collector = ExampleCollector()
        # 3. Create the Inducer
        self.inducer = Inducer(config)
        # 4. Create the Advisor. The Induction Callback needs a reference to the Advisor to force a refresh after induction has finished
        # This is not particularly beautiful
        self.advisor = Advisor(config)
        super().__init__(config, "integrated")

    def _apply_wrappers(self, env, seed):
        env = JoypadSpace(env, self.config["joypad_space"])
        env = MaxAndSkipEnv(env)
        env = DetectObjects(env, detector=self.detector, seed=seed, name=self.name)  # intercept image and convert to object positions
        # translate the bounding boxes to an object/relational representation
        env = PositionObjects(env, positioner=self.positioner,
                              seed=seed, name=self.name)  # intercept image and convert to object positions
        # Track the chosen action. This is necessary for the example callbacks
        env = TrackAction(env, seed=seed, name=self.name)
        env = ResizeAndGrayscale(env)
        env = ImageToPyTorch(env)
        env = BufferWrapper(env, 6)
        env = ScaledFloatFrame(env)
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
