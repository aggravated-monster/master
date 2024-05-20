from abc import ABC

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class IntervalCallback(BaseCallback, ABC):

    def __init__(self, check_freq, verbose=1):
        super(IntervalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.step_logger = Logging.get_logger('game_steps')
        self.step_log_template = "{seed},{total_steps},{episode},{episode_steps},{reward}"

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            rewards = int(self.locals['reward'])

            self.step_logger.info(self.step_log_template.format(seed=self.model.seed,
                                                                total_steps=self.num_timesteps_done,
                                                                episode=self.n_episodes,
                                                                episode_steps=self.locals['episode_step_counter'],
                                                                reward=rewards
                                                                ))

        return True

    def _on_episode(self):
        return True
