from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def update(self, u: float, d: float) -> float:
        pass
