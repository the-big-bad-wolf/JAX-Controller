import yaml
from Bathtub import Bathtub
from Classic import Classic
from CONSENSYS import CONSENSYS

with open("pivotal_parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
    bathtub_params = params["bathtub"]
    A = float(bathtub_params["A"])
    C = float(bathtub_params["C"])
    H0 = float(bathtub_params["H0"])

    plant = Bathtub()
    controller = Classic([1, 2, 3])
    system = CONSENSYS(controller, plant)

    print(A, C, H0)
