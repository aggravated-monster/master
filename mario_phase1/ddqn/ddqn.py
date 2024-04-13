from collections import deque
from typing import Union, List, Any, Optional, Dict

import torch
import numpy as np
import random

from codetiming import Timer
from numpy import NaN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from mario_phase1.callbacks.callback import CallbackList, BaseCallback, DummyCallback
from mario_phase1.ddqn.q_network import QNetwork
from mario_phase1.mario_logging.logging import Logging


class DDQN:

    def __init__(self,
                 env,
                 input_dims,
                 num_actions,
                 lr=0.00025,
                 gamma=0.9,
                 epsilon=1.0,
                 eps_decay=0.99999975,
                 eps_min=0.1,
                 replay_buffer_capacity=50000,
                 batch_size=32,
                 sync_network_rate=10000,
                 verbose=0,
                 seed=None,
                 device: Union[torch.device, str] = "auto",
                 stats_window_size: int = 100
                 ):

        self.env = env
        self.num_actions = num_actions
        self.num_timesteps_done = 0
        self.episode_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = QNetwork(input_dims, num_actions)
        self.target_network = QNetwork(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.f_loss = torch.nn.MSELoss()
        self.n_updates = 0
        # self.f_loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!
        self.loss = NaN

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

        # Administration
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.action_space = env.action_space
        self.stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]

        self.train_logger = Logging.get_logger('train')
        self.step_logger = Logging.get_logger('steps')

        self.set_random_seed(seed)

    def get_env(self):
        return self.env

    def set_random_seed(self, seed):
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        self.set_device_random_seed(seed, using_cuda=self.device == "cuda")
        self.action_space.seed(seed)

    def set_device_random_seed(self, seed: int, using_cuda: bool = False) -> None:
        """
        Taken from stable baselines
        """
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)

        if using_cuda:
            # Deterministic operations for CuDNN, it may impact performances
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `np.array(observation)` instead of `observation`
        # observation is a LIST of numpy arrays because of the LazyFrame wrapper
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
            .unsqueeze(0) \
            .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def sync_networks(self):
        if self.num_timesteps_done % self.sync_network_rate == 0 and self.num_timesteps_done > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()

        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)  # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.f_loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1

        self.decay_epsilon()

        return loss

    def train(self, min_timesteps_to_train: int, callback=None, reset_num_timesteps=True):

        text = str(self.seed) + ";{:0.8f}"

        with Timer(name="Train timer", text=text, logger=self.train_logger.info):

            if self.ep_info_buffer is None or reset_num_timesteps:
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                self.ep_info_buffer = deque(maxlen=self.stats_window_size)
                self.ep_success_buffer = deque(maxlen=self.stats_window_size)

            callback = self._init_callback(callback)
            callback.on_training_start(locals(), globals())

            # I prefer the more deterministic total step count over episode count
            while self.num_timesteps_done < min_timesteps_to_train:

                # Loop over episodes
                done = False
                state, _ = self.env.reset()
                total_reward = 0
                episode_step_counter = 0
                # Mario is done when he reaches the flag, runs out of time or dies
                while not done:
                    # We consider a step as the total processing needed to complete one,
                    # so wrap the whole block in  the timing context manager

                    with Timer(name="Step timer", text=text, logger=self.step_logger.info):
                        action = self.choose_action(state)
                        new_state, reward, done, truncated, info = self.env.step(action)
                        total_reward += reward

                        self.store_in_memory(state, action, reward, new_state, done)
                        self.loss = self.learn()

                        state = new_state
                        self.num_timesteps_done += 1
                        episode_step_counter += 1

                        callback.update_locals(locals())
                        if not callback.on_step():
                            return False

                self.episode_counter += 1
                if not callback.on_episode():
                    return False

                if self.verbose > 0:
                    print("Epsilon:", self.epsilon, "Size of replay buffer:",
                          len(self.replay_buffer), "Total step counter:", self.num_timesteps_done)

            callback.on_training_end()

    def play(self, num_episodes: int, callback=None):

        callback = self._init_callback(callback)
        callback.on_training_start(locals(), globals())

        for i in range(num_episodes):
            done = False
            state, _ = self.env.reset()
            total_reward = 0
            episode_step_counter = 0
            while not done:
                action = self.choose_action(state)
                new_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward

                state = new_state
                self.num_timesteps_done += 1
                episode_step_counter += 1

                callback.update_locals(locals())
                if not callback.on_step():
                    return False

            self.episode_counter += 1
            if not callback.on_episode():
                return False

            if self.verbose > 0:
                print("Episode:", i, " Reward:", total_reward)


    def _init_callback(self, callback):
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = DummyCallback()

        callback.init_callback(self)
        return callback
