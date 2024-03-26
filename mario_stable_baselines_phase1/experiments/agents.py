from codetiming import Timer
from gym.wrappers import ResizeObservation, GrayScaleObservation
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_stable_baselines_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_stable_baselines_phase1.callbacks.episode_callback import EpisodeCallback
from mario_stable_baselines_phase1.callbacks.interval_callback import IntervalCallback
from mario_stable_baselines_phase1.our_logging import our_logging
from mario_stable_baselines_phase1.symbolic_components.detector import Detector
from mario_stable_baselines_phase1.symbolic_components.positioner import Positioner
from mario_stable_baselines_phase1.wrappers.detect_objects import DetectObjects
from mario_stable_baselines_phase1.wrappers.skip_frame import SkipFrame
from mario_stable_baselines_phase1.wrappers.translate_objects import PositionObjects
from master_stable_baselines3 import DQN
from abc import ABC, abstractmethod

from master_stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class TestObject(ABC):

    def __init__(self, config, env, device, tb_log_dir, check_freq, check_dir):

        self.config = config
        self._apply_wrappers(env)
        self.device = device
        self.tb_log_dir = tb_log_dir
        self.checkpointCallback = CheckpointCallback(check_freq=check_freq, save_path=check_dir,
                                                        config=config)
        self.intervalCallback = IntervalCallback(check_freq=1)
        self.episodeCallback = EpisodeCallback()

    def __next_model(self, seed):
        model = DQN(
            "CnnPolicy",
            self.env,
            verbose=1,
            train_freq=1,
            gradient_steps=1,
            gamma=0.9,
            exploration_initial_eps=1.0,
            exploration_fraction=0.2,
            exploration_final_eps=0.1,
            target_update_interval=10000,
            learning_starts=10000,
            buffer_size=100000,
            batch_size=32,
            learning_rate=0.00025,
            policy_kwargs=dict(net_arch=[512, 512]),
            tensorboard_log=self.tb_log_dir,
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
            self.env.reset()
            train_logger = our_logging.Logging.get_logger("train")
            model = self.__next_model(start_seed+n)
            callbacks = self._get_callbacks()
            t = Timer(name="training", logger=train_logger.info, text="{milliseconds:.0f}")
            t.start()
            model.learn(total_timesteps=num_steps, progress_bar=True, callback=callbacks)

            t.stop()


class VanillaAgent(TestObject):
    def __init__(self, config, env, device, tb_log_dir, check_freq, check_dir):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        our_logging.initialize(True, True,"vanilla")

        super().__init__(config, env, device, tb_log_dir, check_freq, check_dir)

    def _apply_wrappers(self, env):
        # 1. Simplify the controls
        env = JoypadSpace(env, RIGHT_ONLY)
        # 2. There is not much difference between frames, so take every fourth
        env = SkipFrame(env, skip=self.config["skip"])  # Num of frames to apply one action to
        # From here on, the observation IS altered again, for efficiency purposes in the RL environment
        env = ResizeObservation(env, shape=84)
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env, keep_dim=True)
        # 5. Wrap inside the Dummy Environment. Standard.
        env = DummyVecEnv([lambda: env])
        # 6. Stack the frames. Standard.
        env = VecFrameStack(env, self.config["stack_size"], channels_order='last')
        self.env = env


class PositionEnabledAgent(TestObject):

    def __init__(self, config, env, device, tb_log_dir, check_freq, check_dir):
        # each new TestObject signifies a new experiments, which we want in separate logfiles.
        our_logging.initialize(True, True, "position_enabled")

        # 1. Create the object detector. This is a YOLO8 model
        self.detector = Detector(config)
        # 2. Create the Translator
        self.positioner = Positioner(config)
        super().__init__(config, env, device, tb_log_dir, check_freq, check_dir)

    def _apply_wrappers(self, env):
        # 1. Simplify the controls
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
        env = ResizeObservation(env, shape=84)
        # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
        env = GrayScaleObservation(env, keep_dim=True)
        # 5. Wrap inside the Dummy Environment. Standard.
        env = DummyVecEnv([lambda: env])
        # 6. Stack the frames. Standard.
        env = VecFrameStack(env, self.config["stack_size"], channels_order='last')

        self.env = env
