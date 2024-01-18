from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, weights: list[float], initital_U: float):
        self.weights = weights
        self.U = initital_U

    @abstractmethod
    def update_weights(self, error_history: list[float]):
        pass

    @abstractmethod
    def update_U(self, E: float, gradient: list[float], sum_epoch_errors: float):
        pass
