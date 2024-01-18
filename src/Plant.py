from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def update(self, U: float, D: float) -> float:
        pass
