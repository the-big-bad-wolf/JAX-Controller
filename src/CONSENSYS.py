from Controller import Controller
from Plant import Plant
import random
import numpy as np


class CONSENSYS:
    def __init__(self, controller: Controller, plant: Plant):
        self.controller = controller
        self.plant = plant
        self.MSE_history: list[float] = []
        pass

    def random_nr(self, min: float, max: float):
        return random.uniform(min, max)

    def MSE(self, error_history: list[float]):
        return np.mean(np.square(error_history)).item()

    def run_epoch(self, timesteps: int):
        errors: list[float] = []
        for _ in range(timesteps):
            error = self.run_timestep()
            errors.append(error)
        return errors

    def run_timestep(self):
        return 1.0

    def start(self, epochs: int, timesteps: int):
        for _ in range(epochs):
            epoch_errors = self.run_epoch(timesteps)
            self.controller.update(epoch_errors)
            MSE = self.MSE(epoch_errors)
            self.MSE_history.append(MSE)
