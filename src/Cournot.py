from Plant import Plant


class Cournot(Plant):
    def __init__(self, p_max: float, cm: float, q_1: float, q_2: float):
        super().__init__([q_1, q_2])
        self.p_max = p_max
        self.cm = cm
        self.q_1 = q_1
        self.q_2 = q_2

    def update(self, u: float, d: float):
        q_1 = self.old_state[0]
        q_2 = self.old_state[1]

        q_1 += u
        q_2 += d

        if q_1 < 0:
            q_1 = 0
        elif q_1 > 1:
            q_1 = 1
        if q_2 < 0:
            q_2 = 0
        elif q_2 > 1:
            q_2 = 1

        q = q_1 + q_2
        p = self.p_max - q
        profit = (p - self.cm) * q_1
        return profit, [q_1, q_2]
