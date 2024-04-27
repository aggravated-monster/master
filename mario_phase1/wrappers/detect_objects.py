from codetiming import Timer

from gym import ObservationWrapper
from pandas import DataFrame

from mario_phase1.mario_logging.logging import Logging


class DetectObjects(ObservationWrapper):
    logger = Logging.get_logger('detect_objects')

    def __init__(self, env, detector, seed=None, name=''):
        super().__init__(env)
        self.detector = detector
        self.detected_objects = None
        self.seed = seed
        self.name = name

    #@Timer(name="DetectObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation) -> DataFrame:

        text = self.name + "," + str(self.seed) + ",{:0.8f}"

        with Timer(name="Detect objects wrapper timer", text=text, logger=self.logger.info):
            positions = self.detector.detect(observation)
            # Store detected objects in wrapper's state. We only need to keep one, which is always the current one.
            self.detected_objects = positions

            # return observation untouched
            return observation
