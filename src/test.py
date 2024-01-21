import jax.numpy as jnp
import jax


def f(x: list[float]):
    return x[0] ** 2 + x[1] ** 2


print(jax.value_and_grad(f)([2.0, 1.0])[1][0])
