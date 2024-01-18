from Controller import Controller


class Classic(Controller):
    def __init__(self, weights: list[float]):
        self.weights = weights

    def update(self, error_history: list[float]):
        pass
