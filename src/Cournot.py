from Plant import Plant
import jax.numpy as jnp


class Cournot(Plant):
    def __init__(self, p_max: float, cm: float, q_1: float, q_2: float):
        super().__init__([q_1, q_2])
        self.p_max = p_max
        self.cm = cm
        self.q_1 = q_1
        self.q_2 = q_2

    def update(self, u: float, d: float):
        q_1 = self.old_state[0]
        q_2 = self.old_state[1]

        q_1 += u
        q_2 += d

        q_1 = jnp.clip(q_1, 0, 1).astype(float)
        q_2 = jnp.clip(q_1, 0, 1).astype(float)

        q = q_1 + q_2
        p = self.p_max - q
        profit = (p - self.cm) * q_1
        return profit, [q_1, q_2]
