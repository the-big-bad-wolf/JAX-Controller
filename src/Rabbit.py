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
        rabbits = self.old_state[0]
        wolfs = self.old_state[1]

        rabbits += (
            self.rabbit_birth_rate * rabbits
            - self.rabbit_eaten_rate * rabbits * wolfs
            + d
        )
        wolfs += (
            self.wolf_food_rate * rabbits * wolfs - self.wolf_death_rate * wolfs + u
        )

        rabbits = max(0, rabbits)
        wolfs = max(0, wolfs)
        return rabbits, [rabbits, wolfs]
