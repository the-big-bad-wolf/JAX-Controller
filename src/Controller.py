from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def __init__(self, weights: list[float], initital_U: float = 0.0):
        self.weights = weights
        self.U = initital_U

    @abstractmethod
    def update(self, error_history: list[float]):
        pass

    @abstractmethod
    def calculate_U(self, E: float, gradient: list[float], sum_epoch_errors: float):
        pass
