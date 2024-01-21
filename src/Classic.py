from Controller import Controller
import jax.numpy as jnp


class Classic(Controller):
    def __init__(self, initital_U: float, weights: list[float], learning_rate: float):
        self.u = initital_U
        self.weights = weights
        self.Kp = weights[0]
        self.Ki = weights[1]
        self.Kd = weights[2]
        self.learning_rate = learning_rate

    def update_weights(self, gradients: list[float]):
        self.weights[0] -= self.learning_rate * gradients[0]
        self.weights[1] -= self.learning_rate * gradients[1]
        self.weights[2] -= self.learning_rate * gradients[2]

    def update_U(self, epoch_errors: list[float], weights: list[float]):
        if len(epoch_errors) < 2:
            return
        Kp = weights[0]
        Ki = weights[1]
        Kd = weights[2]
        self.u = (
            Kp * epoch_errors[-1]
            + Ki * sum(epoch_errors)
            + Kd * (epoch_errors[-1] - epoch_errors[-2])
        )
