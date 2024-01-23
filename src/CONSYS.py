import copy
from Controller import Controller
from Plant import Plant
import random
import jax
import jax.numpy as jnp


class CONSYS:
    def __init__(
        self,
        controller: Controller,
        plant: Plant,
        target: float,
        noise: list[float],
        epochs: int,
        timesteps: int,
    ):
        self.controller = controller
        self.plant = plant
        self.target = target
        self.noise = noise
        self.epochs = epochs
        self.timesteps = timesteps
        self.MSE_history: list[float] = []
        self.param_history: list[jax.Array] = []

    def start(self):
        jax_run_epoch = jax.value_and_grad(self.run_epoch, argnums=0)
        for _ in range(self.epochs):
            plant_copy = copy.deepcopy(self.plant)
            controller_copy = copy.deepcopy(self.controller)
            MSE, gradients = jax_run_epoch(
                self.controller.weights, plant_copy, controller_copy
            )
            print("gradients: ", gradients)
            self.MSE_history.append(MSE)
            self.param_history.append(copy.deepcopy(self.controller.weights))
            self.controller.update_weights(gradients)

    def run_epoch(self, weights: jax.Array, plant: Plant, controller: Controller):
        controller.weights = weights
        noise = self.random_vector()
        errors: jax.Array = jnp.array([])
        for i in range(self.timesteps):
            errors = self.run_timestep(noise[i], errors, plant, controller)
        MSE = self.MSE(errors)
        return MSE

    def run_timestep(
        self,
        d: float,
        errors: jax.Array,
        plant: Plant,
        controller: Controller,
    ):
        u = controller.u
        y = plant.update(u, d)
        error = abs(self.target - y)
        new_errors = jnp.append(errors, error)
        controller.update_U(new_errors, controller.weights)
        return new_errors

    def random_vector(self):
        noise: list[float] = []
        for _ in range(self.timesteps):
            noise.append(random.uniform(self.noise[0], self.noise[1]))
        return noise

    def MSE(self, errors):
        return jnp.mean(jnp.square(errors))
