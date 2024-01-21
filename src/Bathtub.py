from Plant import Plant


class Bathtub(Plant):
    G = 9.81

    def __init__(self, A: float, C: float, H0: float):
        self.A = A
        self.C = C
        self.h = H0

    def update(self, U: float, D: float):
        v = (2 * Bathtub.G * self.h) ** 0.5
        Q = v * self.C
        db = U + D - Q
        dh = db / self.A
        self.h = self.h + dh
        return self.h

    def reset(self):
        pass
