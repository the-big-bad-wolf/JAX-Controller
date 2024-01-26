from Plant import Plant
import jax.numpy as jnp


class Cournot(Plant):
    def __init__(self, p_max: float, cm: float, q_1: float, q_2: float):
        self.p_max = p_max
        self.cm = cm
        self.q_1 = q_1
        self.q_2 = q_2

    def update(self, u: float, d: float):
        self.q_1 += u
        self.q_2 += d

        self.q_1 = jnp.clip(self.q_1, 0, 1).astype(float)
        self.q_2 = jnp.clip(self.q_2, 0, 1).astype(float)

        q = self.q_1 + self.q_2
        p = self.p_max - q
        profit = (p - self.cm) * self.q_1

        self.y = profit.astype(float)
