from Controller import Controller
import jax
import jax.numpy as jnp


class AI(Controller):
    def __init__(
        self,
        initital_U: float,
        weights: jax.Array,
        biases: jax.Array,
        layers: int,
        neurons_per_layer: int,
        activation_function,
        learning_rate: float,
    ):
        super().__init__(weights, initital_U, learning_rate)
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.biases = biases

    def update_weights(self, gradients: jax.Array):
        pass

    def update_U(self, errors: jax.Array, weights: jax.Array):
        # Calculate the neural network
        for layer in range(self.layers):
            # Apply activation function to the weighted sum of inputs
            weighted_sum = jnp.dot(errors, weights[layer]) + self.biases[layer]
            self.u = self.activation_function(weighted_sum)
