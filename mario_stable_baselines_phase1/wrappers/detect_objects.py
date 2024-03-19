from codetiming import Timer

from gym import ObservationWrapper
from pandas import DataFrame

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class DetectObjects(ObservationWrapper):
    logger = Logging.get_logger('detection')

    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector

    @Timer(name="DetectObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation) -> DataFrame:
        positions = self.detector.detect(observation)

        return positions