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
        self.weights = [
            (w - self.learning_rate * dw, b - self.learning_rate * db)
            for (w, b), (dw, db) in zip(self.weights, gradients)
        ]

    def calculate_U(self, errors: jax.Array, weights_and_bias: jax.Array):
        last_error = errors[-1]
        sum_error = jnp.sum(errors)
        derivative_error = errors[-1] - errors[-2]
        features = jnp.array([last_error, sum_error, derivative_error])
        activations = features

        for i in range(weights_and_bias):
            weights, biases = weights_and_bias[i]
            if i == len(weights_and_bias) - 1:
                activations = jnp.dot(activations, weights) + biases
            else:
                activations = self.activation_function(
                    jnp.dot(activations, weights) + biases
                )
        return activations
