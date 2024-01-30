from Plant import Plant


class Rabbit(Plant):
    def __init__(
        self,
        initial_rabbits: float,
        rabbit_birth_rate: float,
        rabbit_eaten_rate: float,
        initial_wolves: float,
        wolf_death_rate: float,
        wolf_food_rate: float,
    ):
        self.rabbits = initial_rabbits
        self.wolves = initial_wolves
        self.rabbit_birth_rate = rabbit_birth_rate
        self.rabbit_eaten_rate = rabbit_eaten_rate
        self.wolf_death_rate = wolf_death_rate
        self.wolf_food_rate = wolf_food_rate

    def update(self, u: float, d: float):
        rabbit_increase = (
            self.rabbit_birth_rate * self.rabbits
            - self.rabbit_eaten_rate * self.rabbits * self.wolves
            + u
        )
        wolves_increase = (
            self.wolf_food_rate * self.rabbits * self.wolves
            - self.wolf_death_rate * self.wolves
            + d
        )
        self.rabbits = max(0, self.rabbits + rabbit_increase)
        self.wolves = max(0, self.wolves + wolves_increase)

        self.y = self.rabbits
