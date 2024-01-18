from Controller import Controller
from Plant import Plant
import random
import numpy as np


class CONSENSYS:
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
        self.MSE_history: list[float] = []
        pass

    def random_vector(self, min: float, max: float, timesteps: int):
        noise: list[float] = []
        for _ in range(timesteps):
            noise.append(random.uniform(min, max))
        return noise

    def MSE(self, error_history: list[float]):
        return np.mean(np.square(error_history)).item()

    def run_epoch(self, timesteps: int):
        noise = self.random_vector(self.noise[0], self.noise[1], timesteps)
        errors: list[float] = []
        for i in range(timesteps):
            error = self.run_timestep(noise[i])
            errors.append(error)
        return errors

    def run_timestep(self, D: float):
        U = self.controller.U
        Y = self.plant.update(U, D)
        error = abs(self.target - Y)
        self.controller.calculate_U(0, [0], 0)
        return error

    def start(self, epochs: int, timesteps: int):
        for _ in range(epochs):
            epoch_errors = self.run_epoch(timesteps)
            self.controller.update(epoch_errors)
            MSE = self.MSE(epoch_errors)
            self.MSE_history.append(MSE)
