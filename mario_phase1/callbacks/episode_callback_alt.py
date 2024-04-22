from abc import ABC

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class EpisodeCallbackAlt(BaseCallback, ABC):

    def __init__(self):
        super(EpisodeCallbackAlt, self).__init__()
        # Logging
        self.episode_logger = Logging.get_logger('game_episodes')
        self.console_logger = Logging.get_logger('console')
        self.console_log_template = "Episode {episode} finished after {episode_steps} at timestep {steps} with a total reward of {episode_reward} and loss {loss}."
        self.episode_log_template = "{seed},{total_steps},{episode},{episode_steps},{episode_reward},{distance},{velocity},{time},{score},{flag},{loss},{epsilon}"

    def _on_episode(self) -> bool:
        distance = self.locals['info']['x_pos']
        time = self.locals['info']['time']
        score = self.locals['info']['score']
        flag = self.locals['info']['flag_get']

        # Q: begint dat spel altijd met 400 en is het terugtellen monotoon?
        velocity = distance / (400 - time)

        episode_steps = self.locals['episode_step_counter']
        episode_reward = self.locals['total_reward']

        self.episode_logger.info(self.episode_log_template.format(seed=self.model.seed,
                                                                  total_steps=self.num_timesteps_done,
                                                                  episode=self.n_episodes,
                                                                  episode_steps=episode_steps,
                                                                  episode_reward=episode_reward,
                                                                  distance=distance,
                                                                  velocity=velocity,
                                                                  time=time,
                                                                  score=score,
                                                                  flag=flag,
                                                                  loss=self.model.loss,
                                                                  epsilon=self.model.epsilon
                                                                  ))
        self.console_logger.info(self.console_log_template.format(episode=self.n_episodes,
                                                                  episode_steps=episode_steps,
                                                                  steps=self.num_timesteps_done,
                                                                  episode_reward=episode_reward,
                                                                  loss=self.model.loss
                                                                  ))


        return True

    def _on_step(self):
        return True
