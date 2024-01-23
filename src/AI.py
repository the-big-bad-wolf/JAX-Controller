from Controller import Controller
import jax
import jax.numpy as jnp


class AI(Controller):
    def __init__(
        self,
        initital_U: float,
        weights: jax.Array,
        learning_rate: float,
        layers: int,
        neurons_per_layer: int,
        activation_function,
    ):
        super().__init__(weights, initital_U, learning_rate)
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function

    def update_weights(self, gradients: jax.Array):
        pass

    def update_U(self, errors: jax.Array, weights: jax.Array):
        pass
