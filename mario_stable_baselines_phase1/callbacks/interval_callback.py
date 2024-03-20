import json
import os
from master_stable_baselines3.common.callbacks import BaseCallback
from master_stable_baselines3.common.logger import TensorBoardOutputFormat

from mario_stable_baselines_phase1.our_logging.our_logging import Logging



class IntervalCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def __init__(self, check_freq, verbose=1):
        super(IntervalCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            rewards = self.locals['rewards']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("reward/step/env #{}".format(i + 1),
                                                    rewards[i],
                                                    self.n_calls)

        return True