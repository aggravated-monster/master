from gym import ObservationWrapper
from mario_DQN_baseline.our_logging.our_logging import Logging
from pandas import DataFrame
from codetiming import Timer
class DetectObjects(ObservationWrapper):
    logger = Logging.get_logger('detection')

    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector

    @Timer(name="DetectObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation) -> DataFrame:
        positions = self.detector.detect(observation)

        return positions
