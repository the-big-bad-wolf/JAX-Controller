from Plant import Plant


class Bathtub(Plant):
    G = 9.81

    def __init__(self, A: float, C: float, H0: float):
        super().__init__(H0)
        self.A = A
        self.C = C

    def update(self, u: float, d: float):
        v = (2 * Bathtub.G * self.old_state) ** 0.5
        q = v * self.C
        db = u + d - q
        dh = db / self.A
        y = self.old_state + dh
        y = max(y, 0)
        return y, y
