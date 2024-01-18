from Controller import Controller


class Classic(Controller):
    def update_weights(self, error_history: list[float]):
        pass

    def update_U(self, E: float, gradient: list[float], sum_epoch_errors: float):
        pass
