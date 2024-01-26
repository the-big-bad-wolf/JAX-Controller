from abc import ABC, abstractmethod
import jax


class Controller(ABC):
    def __init__(self, weights: jax.Array, initital_U: float, learning_rate: float):
        self.weights = weights
        self.u = initital_U
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(self, errors: jax.Array):
        pass

    @abstractmethod
    def update_U(self, errors: jax.Array, weights: jax.Array):
        pass
