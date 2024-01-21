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
        self.param_history = []

    def random_vector(self, min: float, max: float, timesteps: int):
        noise: list[float] = []
        for _ in range(timesteps):
            noise.append(random.uniform(min, max))
        return noise

    def MSE(self, error_history):
        temp = []
        for i in range(len(error_history)):
            temp.append(error_history[i] ** 2)
        sum = 0
        for square in temp:
            if jnp.isnan(square):
                continue
            sum += square
        return sum / len(error_history)

    def run_epoch(self, weights: list[float], plant: Plant, controller: Controller):
        controller.weights = weights
        noise = self.random_vector(self.noise[0], self.noise[1], self.timesteps)
        errors = jnp.array([])
        for i in range(self.timesteps):
            error = self.run_timestep(noise[i], errors, plant, controller)
            errors = jnp.append(errors, error)
        MSE = self.MSE(errors)
        return MSE

    def run_timestep(
        self,
        D: float,
        errors,
        plant: Plant,
        controller: Controller,
    ):
        u = controller.u
        y = plant.update(u, D)
        error = abs(self.target - y)
        controller.update_U(errors, controller.weights)
        return error

    def start(self):
        gradfunc = jax.value_and_grad(self.run_epoch, argnums=0)
        for _ in range(self.epochs):
            plant_copy = copy.deepcopy(self.plant)
            controller_copy = copy.deepcopy(self.controller)
            MSE, gradients = gradfunc(
                self.controller.weights, plant_copy, controller_copy
            )
            self.MSE_history.append(MSE)
            self.param_history.append(copy.deepcopy(self.controller.weights))
            self.controller.update_weights(gradients)
