from Plant import Plant


class Rabbit(Plant):
    def __init__(
        self,
        rabbit_birth_rate: float,
        rabbit_eaten_rate: float,
        initial_rabbits: float,
        wolf_death_rate: float,
        wolf_food_rate: float,
        initial_wolves: float,
    ):
        super().__init__([initial_rabbits, initial_wolves])
        self.rabbit_birth_rate = rabbit_birth_rate
        self.rabbit_eaten_rate = rabbit_eaten_rate
        self.wolf_death_rate = wolf_death_rate
        self.wolf_food_rate = wolf_food_rate

    def update(self, u: float, d: float):
        old_rabbits = self.old_state[0]
        old_wolves = self.old_state[1]

        new_rabbits = old_rabbits + (
            self.rabbit_birth_rate * old_rabbits
            - self.rabbit_eaten_rate * old_rabbits * old_wolves
            + u
        )
        new_wolves = old_wolves + (
            self.wolf_food_rate * old_rabbits * old_wolves
            - self.wolf_death_rate * old_wolves
            + d
        )
        new_rabbits = max(0, new_rabbits)
        new_wolves = max(0, new_wolves)
        return new_rabbits, [new_rabbits, new_wolves]
