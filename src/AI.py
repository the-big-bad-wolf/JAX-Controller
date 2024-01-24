from Controller import Controller
import jax
import jax.numpy as jnp


class AI(Controller):
    def __init__(
        self,
        initital_U: float,
        weights,
        activation_function,
        learning_rate: float,
    ):
        self.u = initital_U
        self.learning_rate = learning_rate
        self.weights = weights
        self.activation_function = activation_function

    def update_weights(self, gradients: jax.Array):
        pass

    def calculate_U(self, errors: jax.Array, weights: jax.Array):
        pass
