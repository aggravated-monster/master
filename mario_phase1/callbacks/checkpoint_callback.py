import json
import os
from abc import ABC

import torch

from mario_phase1.callbacks.callback import BaseCallback


class CheckpointCallback(BaseCallback, ABC):

    def __init__(self, check_freq, save_path, config, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        # keep track of the configuration used for a training session
        self.config = config

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path + "/configuration.json", "w") as outfile:
                json.dump(self.config, outfile, indent=4)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.save_model(os.path.join(self.save_path, "model_" + str(self.num_timesteps_done) + "_iter.pt"))

        return True

    def _on_episode(self):
        return True

    def save_model(self, path):
        torch.save(self.model.online_network.state_dict(), path)
