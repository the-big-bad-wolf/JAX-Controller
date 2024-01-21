from Controller import Controller
import jax
import jax.numpy as jnp


class Classic(Controller):
    def __init__(self, initital_U: float, weights: jax.Array, learning_rate: float):
        super().__init__(weights, initital_U, learning_rate)
        self.Kp = weights[0]
        self.Ki = weights[1]
        self.Kd = weights[2]

    def update_weights(self, gradients: list[float]):
        self.Kp -= self.learning_rate * gradients[0]  # Kp
        self.Ki -= self.learning_rate * gradients[1]  # Ki
        self.Kd -= self.learning_rate * gradients[2]  # Kd
        self.weights = jnp.array([self.Kp, self.Ki, self.Kd])

    def update_U(self, errors: jax.Array, weights: list[float]):
        if len(errors) < 2:
            return
        Kp = weights[0]
        Ki = weights[1]
        Kd = weights[2]
        self.u = Kp * errors[-1] + Ki * sum(errors) + Kd * (errors[-1] - errors[-2])
