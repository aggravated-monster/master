from gym import Wrapper, ObservationWrapper
from gym.wrappers import FrameStack
from gym.error import DependencyNotInstalled
from numpy import ndarray
from pandas import DataFrame
from to_delete.mario_phase1_youtube_vanilla.utils import get_current_date_time_string


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info

class CaptureFrames(ObservationWrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.env_name = env_name

    def observation(self, observation):
        try:
            import cv2
        except ImportError: raise DependencyNotInstalled(
            "opencv is not installed, run 'pip install gym[other]'")
        cv2.imshow('game', observation)
        cv2.imwrite('./frames/' + self.env_name + 'img_' + self.env_name + '_' + get_current_date_time_string() + '.png', observation)

class DetectObjects(ObservationWrapper):
    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector

    def observation(self, observation) -> DataFrame:
        """Updates the observations by detecting the objects in the image.

        Args:
            observation: The observation to use for detection

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise DependencyNotInstalled(
                "opencv is not install, run `pip install gym[other]`"
            )

        cv2.imwrite('./img_plaatje.png', observation)

        positions = self.detector.detect(observation)
        # print dataframe.
        # print(positions)

        return positions


class TransformAndResize(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape

    def observation(self, observation) -> ndarray:
        """Transforms the observation to a ndarray of shape self.shape.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        # expecting input to be a DataFrame with first column 'name'.
        # this column can be dropped. It will only be relevant in Phase2
        positions = observation.drop(['name'], axis=1).to_numpy().copy()

        # resize to the original 84*84, to not break the pipeline, until we replace the Agent's CNN
        # resize on the object pads the missing numbers with zeros
        positions.resize(self.shape, refcheck=False)
        # print(positions)
        # print(positions.shape)

        return positions


def apply_wrappers(env, detector):
    # NOTE: adding/removing a wrapper has repercussions for generate_clips.py
    # Search for 'env.env.' and make sure the number of env fit the number of wrappers to be
    # traversed to get to the SkipFrame wrapper
    env = SkipFrame(env, skip=4)  # Num of frames to apply one action to
    # I noticed that in the resized image, Mario looks very similar to the floor, which may be a problem
    # for object detection. I'm keeping the wrapper for now to not break the pipeline, but
    # we probably don't need to resize the image
    #env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
    #env = GrayScaleObservation(env)
    env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    env = TransformAndResize(env, shape=(5, 16))
    env = FrameStack(env, num_stack=4, lz4_compress=True)  # May need to change lz4_compress to False if issues arise
    return env


def apply_img_capture_wrappers(env, env_name):
    # env = SkipFrame(env, skip=4)  # Num of frames to apply one action to
    env = CaptureFrames(env, env_name)  # intercept image and convert to object positions

    return env