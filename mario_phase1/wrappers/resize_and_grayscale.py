import cv2
import gym
import numpy as np


class ResizeAndGrayscale(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    And applies semantic    def learn(self):
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

        return loss segmentation if set to. Otherwise uses grayscale normal frames.
    Returns numpy array
    """

    def __init__(self, env=None):
        super(ResizeAndGrayscale, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ResizeAndGrayscale.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img_og = np.reshape(frame, [240, 256, 3]).astype(np.uint8)
            img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        else:
            assert False, "Unknown resolution."

        # Re-scale image to fit model.
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])

        return x_t.astype(np.uint8)