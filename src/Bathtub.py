from Plant import Plant


class Bathtub(Plant):
    G = 9.81

    def __init__(self, A: float, C: float, H0: float):
        self.h = H0
        self.A = A
        self.C = C

    def update(self, u: float, d: float):
        v = (2 * Bathtub.G * self.h) ** 0.5
        q = v * self.C
        db = u + d - q
        dh = db / self.A
        self.h += dh
        self.h = max(self.h, 0)

        self.y = self.h
