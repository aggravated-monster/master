from codetiming import Timer

from gym import ObservationWrapper
from pandas import DataFrame

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class DetectObjects(ObservationWrapper):
    logger = Logging.get_logger('detection')

    def __init__(self, env, detector, seed=None):
        super().__init__(env)
        self.detector = detector
        self.detected_objects = None
        self.seed = seed

    #@Timer(name="DetectObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation) -> DataFrame:

        text = str(self.seed) + ";{:0.8f}"

        with Timer(name="ChooseAction wrapper timer", text=text, logger=self.logger.info):
            positions = self.detector.detect(observation)
            # Store detected objects in wrapper's state. We only need to keep one, which is always the current one.
            self.detected_objects = positions

            # return observation untouched
            return observation
