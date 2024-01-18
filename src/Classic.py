from Controller import Controller


class Classic(Controller):
    def __init__(self, weights: list[float]):
        self.weights = weights

    def update(self, error_history: list[float]):
        pass

    def calculate_U(self, E: float, gradient: list[float], sum_epoch_errors: float):
        pass
