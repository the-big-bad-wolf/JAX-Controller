from abc import ABC, abstractmethod
from typing import Any, Tuple


class Plant(ABC):
    def __init__(self, old_state):
        self.old_state = old_state

    @abstractmethod
    def update(self, u: float, d: float) -> Tuple[Any, Any]:
        pass
