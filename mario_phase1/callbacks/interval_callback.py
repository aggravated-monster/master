from abc import ABC

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class IntervalCallback(BaseCallback, ABC):

    def __init__(self, check_freq, verbose=1):
        super(IntervalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.step_logger = Logging.get_logger('steps')
        self.step_log_template = "{step},{reward}"

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            rewards = self.locals['reward']

            self.step_logger.info(self.step_log_template.format(step=self.num_timesteps_done,
                                                                reward=rewards
                                                                ))

        return True

    def _on_episode(self):
        return True
