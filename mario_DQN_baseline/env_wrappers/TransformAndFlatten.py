
from gym import ObservationWrapper
import numpy as np
from numpy import ndarray


class TransformAndFlatten(ObservationWrapper):
    def __init__(self, env, dim):
        super().__init__(env)
        self.dim = dim

    def observation(self, observation) -> ndarray:
        """Transforms the observation to a ndarray of shape self.dim.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        # expecting input to be a DataFrame with first column 'name'.
        # this column can be dropped. It will only be relevant in Phase2
        positions = observation.drop(['name'], axis=1).to_numpy().copy()

        # make a 1D vector that fits the mlpPolicy
        flattened = positions.reshape(-1)
        # padding the array with negative 1
        padded = np.pad(flattened, (0, self.dim - (positions.shape[0] * positions.shape[1])), 'constant',
                        constant_values=(-1))

        return padded