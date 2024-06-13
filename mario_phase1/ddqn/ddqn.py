#Code modified from Javier Montalvo (https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning).


import random

import numpy as np
import torch
import torch.nn as nn
from codetiming import Timer

from mario_phase1.callbacks.callback import CallbackList, BaseCallback, DummyCallback
from mario_phase1.ddqn.q_network import DQNSolver
from mario_phase1.mario_logging.logging import Logging, RIGHT_ONLY_HUMAN


class DQNAgent:

    def __init__(self,
                 env,
                 input_dims,
                 num_actions,
                 max_memory_size,
                 batch_size,
                 gamma,
                 lr,
                 dropout,
                 exploration_max,
                 exploration_min,
                 exploration_decay,
                 pretrained,
                 verbose=1,
                 seed=None,
                 advisor=None,
                 name="",
                 choose_intrusive=False,
                 ):

        # Define DQN Layers
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.local_net = DQNSolver(input_dims, num_actions).to(self.device)
        self.target_net = DQNSolver(input_dims, num_actions).to(self.device)

        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 1000  # Copy the local model weights into the target network every 1000 steps
        self.step = 0
        self.n_updates = 0

        self.env = env
        self.num_timesteps_done = 0
        self.episode_counter = 0

        # Reserve memory for the experience replay "dataset"
        self.max_memory_size = max_memory_size

        self.STATE_MEM = torch.zeros(max_memory_size, *self.input_dims)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, *self.input_dims)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Set up agent learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Huber loss
        self.exploration_max = exploration_max
        self.epsilon = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.loss = np.NaN

        # Administration
        self.verbose = verbose
        self.seed = seed
        self.name = name

        self.train_logger = Logging.get_logger('train')
        self.step_logger = Logging.get_logger('steps')
        self.action_logger = Logging.get_logger('choose_action_ddqn')
        self.advice_logger = Logging.get_logger('advice_given_ddqn')
        self.advice_log_template = "{timestep},{advice};{action_chosen};{state}"

        self.set_random_seed(seed)

        # Advisor
        self.advisor = advisor
        self.choose_intrusive = choose_intrusive


    def set_random_seed(self, seed):
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        self.set_device_random_seed(seed, using_cuda=self.device == "cuda")
        self.env.action_space.seed(seed)

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


    def get_env(self):
        return self.env

    def remember(self, state, action, reward, state2, done):  # Store "remembrance" on experience replay
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        # Randomly sample 'batch size' experiences from the experience replay
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]

        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        self.step += 1

        # Epsilon-greedy action block now also used for advice
        if random.random() < self.epsilon:
            if self.advisor is None:
                # explore
                return torch.tensor([[random.randrange(self.num_actions)]])
            else:
                text = self.name + "," + str(self.seed) + ",{:0.8f}"
                with Timer(name="ChooseAction wrapper timer", text=text, logger=self.action_logger.info):
                    advice = self.__ask_advice()
                    if advice is not None:
                        return torch.tensor([[advice]])
                    # else, explore
                    return torch.tensor([[random.randrange(self.num_actions)]])

        # Local net is used for the policy
        logits = self.local_net(state.to(self.device))

        action = torch.argmax(logits).unsqueeze(0).unsqueeze(0).cpu()

        return action


    def act_intrusive(self, state):
        self.step += 1

        # Epsilon-greedy action block also used for advice
        if random.random() < self.epsilon:
            if self.advisor is None:
                # explore
                return torch.tensor([[random.randrange(self.num_actions)]])
            else:
                text = self.name + "," + str(self.seed) + ",{:0.8f}"
                with Timer(name="ChooseAction wrapper timer", text=text, logger=self.action_logger.info):
                    advice = self.__ask_advice()
                    if advice is not None:
                        return torch.tensor([[advice]])
                    # else, explore
                    return torch.tensor([[random.randrange(self.num_actions)]])

        # But in the intrusive variant, the non-epsilon block is also used for advice
        # Local net is used for the policy
        logits = self.local_net(state.to(self.device))
        action = torch.argmax(logits).unsqueeze(0).unsqueeze(0).cpu()
        # Maybe override this choice by selecting a better one
        if self.advisor is not None:
            text = self.name + "," + str(self.seed) + ",{:0.8f}"
            with Timer(name="ChooseAction wrapper timer", text=text, logger=self.action_logger.info):
                advice = self.__ask_advice()
                if advice is not None:
                    return torch.tensor([[advice]])

        return action


    def __ask_advice(self):
        current_facts = " ".join(self.env.relevant_positions[0][1])
        advice = self.advisor.advise(current_facts)
        if advice is None:
            advice = "no model"
            # if Advisor returns None, no model was found
            # Proceed with caution.
            action_chosen = None
            #action_chosen = np.random.randint(self.num_actions)

        elif len(advice) == 0: # advice = []
            # no advice found.
            action_chosen = np.random.randint(self.num_actions)
            #action_chosen = None

        else:
            # advice found
            # given that this might be a list, choose one
            action_chosen = np.random.choice(advice)
            # convert to correct index
            action_chosen = RIGHT_ONLY_HUMAN.index(action_chosen)
            advice = " ".join(advice)

        # log the things
        if action_chosen is None:
            action_chosen_str = "None"
        else:
            action_chosen_str = RIGHT_ONLY_HUMAN[action_chosen]

        self.__log_advice(str(self.num_timesteps_done+1), advice, action_chosen_str, current_facts)

        return action_chosen

    def __log_advice(self, timestep, advice, action_chosen, observation):
        self.advice_logger.info(self.advice_log_template.format(timestep=timestep,
                                                                advice=advice,
                                                                action_chosen=action_chosen,
                                                                state=observation
                                                                ))

    def copy_model(self):
        # Copy local net weights into target netsprint :- not close.
        self.target_net.load_state_dict(self.local_net.state_dict())

    def learn(self):
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = REWARD + torch.mul((self.gamma *
                                     self.target_net(STATE2).max(1).values.unsqueeze(1)),
                                    1 - DONE)
        current = self.local_net(STATE).gather(1, ACTION.long())  # Local net approximation of Q-value

        loss = self.l1(current, target)
        loss.backward()  # Compute gradientsaction_space
        self.optimizer.step()  # Backpropagate error

        self.epsilon *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.epsilon = max(self.epsilon, self.exploration_min)

        self.n_updates += 1

        return loss

    def load_model(self, path):
        self.local_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


    def train(self, min_timesteps_to_train: int, callback=None, reset_num_timesteps=True):

        text = self.name + "," + str(self.seed) + ",{:0.8f}"

        with Timer(name="Train timer", text=text, logger=self.train_logger.info):

            total_rewards = []
            ending_positions = []

            callback = self._init_callback(callback)
            callback.on_training_start(locals(), globals())

            # I prefer the more deterministic total step count over episode count
            while self.num_timesteps_done < min_timesteps_to_train:

                # Loop over episodes
                done = False
                # Reset state and convert to tensor
                state, _ = self.env.reset()
                state = torch.Tensor(np.array([state]))
                total_reward = 0
                episode_step_counter = 0
                # Mario is done when he reaches the flag, runs out of time or dies
                while not done:
                    # We consider a step as the total processing needed to complete one,
                    # so wrap the whole block in  the timing context manager

                    with Timer(name="Step timer", text=text, logger=self.step_logger.info):
                        if self.choose_intrusive:
                            action = self.act_intrusive(state)
                        else:
                            action = self.act(state)
                        new_state, reward, done, truncated, info = self.env.step(int(action[0]))
                        total_reward += reward

                        # Change to next state
                        new_state = torch.Tensor(np.array([new_state]))
                        # Change reward type to tensor (to store in ER)
                        reward = torch.tensor(np.array([reward])).unsqueeze(0)
                        # Is the new state a terminal state?
                        done = torch.tensor(np.array([int(done)])).unsqueeze(0)

                        self.remember(state, action, reward, new_state, done)
                        self.loss = self.learn()

                        state = new_state
                        self.num_timesteps_done += 1
                        episode_step_counter += 1

                        callback.update_locals(locals())
                        if not callback.on_step():
                            return False

                # Epside has finished
                self.episode_counter += 1

                if not callback.on_episode():
                    return False

                if self.verbose > 0:
                    print("Epsilon:", self.epsilon, "Size of replay buffer:",
                          self.num_in_queue, "Total step counter:", self.num_timesteps_done)


            callback.on_training_end()


    def train_episodes(self, num_episodes: int, callback=None, reset_num_timesteps=True):

        text = self.name + "," + str(self.seed) + ",{:0.8f}"

        with Timer(name="Train timer", text=text, logger=self.train_logger.info):

            total_rewards = []
            ending_positions = []

            callback = self._init_callback(callback)
            callback.on_training_start(locals(), globals())

            while self.episode_counter < num_episodes:

                # Loop over episodes
                done = False
                # Reset state and convert to tensor
                state, _ = self.env.reset()
                state = torch.Tensor(np.array([state]))
                total_reward = 0
                episode_step_counter = 0
                # Mario is done when he reaches the flag, runs out of time or dies
                while not done:
                    # We consider a step as the total processing needed to complete one,
                    # so wrap the whole block in  the timing context manager

                    with Timer(name="Step timer", text=text, logger=self.step_logger.info):
                        if self.choose_intrusive:
                            action = self.act_intrusive(state)
                        else:
                            action = self.act(state)
                        new_state, reward, done, truncated, info = self.env.step(int(action[0]))
                        total_reward += reward

                        # Change to next state
                        new_state = torch.Tensor(np.array([new_state]))
                        # Change reward type to tensor (to store in ER)
                        reward = torch.tensor(np.array([reward])).unsqueeze(0)
                        # Is the new state a terminal state?
                        done = torch.tensor(np.array([int(done)])).unsqueeze(0)

                        self.remember(state, action, reward, new_state, done)
                        self.loss = self.learn()

                        state = new_state
                        self.num_timesteps_done += 1
                        episode_step_counter += 1

                        callback.update_locals(locals())
                        if not callback.on_step():
                            return False

                # Epside has finished
                self.episode_counter += 1

                if not callback.on_episode():
                    return False

                if self.verbose > 0:
                    print("Epsilon:", self.epsilon, "Size of replay buffer:",
                          self.num_in_queue, "Total step counter:", self.num_timesteps_done)


            callback.on_training_end()


    def play(self, num_episodes: int, callback=None):

        callback = self._init_callback(callback)
        callback.on_training_start(locals(), globals())

        for i in range(num_episodes):
            done = False
            state, _ = self.env.reset()
            state = torch.Tensor(np.array([state]))
            total_reward = 0
            episode_step_counter = 0
            while not done:
                action = self.act(state)
                new_state, reward, done, truncated, info = self.env.step(int(action[0]))
                total_reward += reward

                # Change to next state
                new_state = torch.Tensor(np.array([new_state]))
                # Change reward type to tensor (to store in ER)
                reward = torch.tensor(np.array([reward])).unsqueeze(0)
                # Is the new state a terminal state?
                done = torch.tensor(np.array([int(done)])).unsqueeze(0)

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