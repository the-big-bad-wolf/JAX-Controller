from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def __init__(self, weights: list[float]):
        self.weights = weights

    @abstractmethod
    def update(self, error_history: list[float]):
        pass
