import json
import os
import pickle
from abc import ABC

import torch
from matplotlib import pyplot as plt

from mario_phase1.callbacks.callback import BaseCallback


class CheckpointCallbackAlt(BaseCallback, ABC):

    def __init__(self, config, name='', verbose=1):
        super(CheckpointCallbackAlt, self).__init__(verbose)
        self.check_freq = config["checkpoint_frequency"]
        self.save_path = config["checkpoint_dir"]
        # keep track of the configuration used for a training session
        self.config = config
        self.name = name

        # Store rewards and positions
        self.total_rewards = []
        self.ending_positions = []

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path + "/configuration.json", "w") as outfile:
                json.dump(self.config, outfile, indent=4)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.save_model()

        return True

    def _on_episode(self):
        episode_reward = self.locals['total_reward']
        self.total_rewards.append(episode_reward)
        self.ending_positions.append(self.model.ending_position)
        return True

    def _on_training_end(self) -> None:
        # Total number of timesteps reached
        # Training has finished
        # This stuff is also callback stuff. Move
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot(self.total_rewards)
        plt.show()

    def save_model(self):
        #with open(self.save_path + "ending_position.pkl", "wb") as f:
        #    pickle.dump(self.model.ending_position, f)
        #with open(self.save_path + "num_in_queue.pkl", "wb") as f:
        #    pickle.dump(self.model.num_in_queue, f)
        #with open(self.save_path + self.name + "_total_rewards.pkl", "wb") as f:
        #    pickle.dump(self.total_rewards, f)
        #with open(self.save_path + self.name + "_ending_positions.pkl", "wb") as f:
        #    pickle.dump(self.ending_positions, f)

        torch.save(self.model.local_net.state_dict(), self.save_path + str(self.num_timesteps_done) +
                   "_" + str(self.model.seed) + "_online_net.pt")
        torch.save(self.model.target_net.state_dict(), self.save_path + str(self.num_timesteps_done) +
                   "_" + str(self.model.seed) + "_target_net.pt")

        if self.config["save_replay_buffer"]:  # If save experience replay is on.
            print("Saving Experience Replay....")
            torch.save(self.model.STATE_MEM, self.save_path + "STATE_MEM.pt")
            torch.save(self.model.ACTION_MEM, self.save_path + "ACTION_MEM.pt")
            torch.save(self.model.REWARD_MEM, self.save_path + "REWARD_MEM.pt")
            torch.save(self.model.STATE2_MEM, self.save_path + "STATE2_MEM.pt")
            torch.save(self.model.DONE_MEM, self.save_path + "DONE_MEM.pt")
