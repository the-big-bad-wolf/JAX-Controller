from Plant import Plant


class Cournot(Plant):
    def __init__(self, p_max: float, cm: float, q_1: float, q_2: float):
        self.p_max = p_max
        self.cm = cm
        self.q_1 = q_1
        self.q_2 = q_2

    def update(self, u: float, d: float):
        self.q_1 += u
        self.q_2 += d
        if self.q_1 < 0:
            self.q_1 = 0
        if self.q_2 < 0:
            self.q_2 = 0
        if self.q_1 > 1:
            self.q_1 = 1
        if self.q_2 > 1:
            self.q_2 = 1

        q = self.q_1 + self.q_2
        p = self.p_max - q
        profit = (p - self.cm) * self.q_1
        print("profit: ", profit, "q1: ", self.q_1, "q2: ", self.q_2)
        return profit
