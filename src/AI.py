from Controller import Controller
import jax
import jax.numpy as jnp


class AI(Controller):
    def __init__(
        self,
        initital_U: float,
        weights,
        layers: int,
        neurons_per_layer: int,
        activation_function,
        learning_rate: float,
    ):
        self.u = initital_U
        self.learning_rate = learning_rate
        self.weights = weights
        self.first_hidden_weights = weights[0]
        self.hidden_weights = weights[1]
        self.output_weights = weights[2]
        self.biases = weights[3]
        self.output_bias = weights[4]
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function

    def update_weights(self, gradients: jax.Array):
        pass

    def update_U(self, errors: jax.Array, weights: jax.Array):
        a = jnp.array([errors[-1], jnp.sum(errors), errors[-1] - errors[-2]])

        a = jnp.dot(self.first_hidden_weights, a)
        a += self.biases[0]
        a = self.activation_function(a)

        for i in range(0, self.layers - 1):
            a = jnp.dot(self.hidden_weights[i], a)
            a += self.biases[i + 1]
            a = self.activation_function(a)

        a = jnp.dot(self.output_weights, a)
        a += self.output_bias
        a = self.activation_function(a)
        print(a)
