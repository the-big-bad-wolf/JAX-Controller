import yaml
from Bathtub import Bathtub
from Classic import Classic
from CONSYS import CONSYS

with open("pivotal_parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
    plant = str(params["plant"])
    controller = str(params["controller"])
    noise = params["noise"]
    epochs = int(params["epochs"])
    timesteps = int(params["timesteps"])

    match controller:
        case "classic":
            classic_params = params["classic"]
            kp = float(classic_params["Kp"])
            ki = float(classic_params["Ki"])
            kd = float(classic_params["Kd"])
            controller = Classic([kp, ki, kd], 0)
        case _:
            print("No valid controller")
            exit()

    match plant:
        case "bathtub":
            bathtub_params = params["bathtub"]
            A = float(bathtub_params["A"])
            C = float(bathtub_params["C"])
            target = float(bathtub_params["H0"])
            plant = Bathtub(A, C, target)
        case _:
            print("No valid plant.")
            exit()

    system = CONSYS(controller, plant, target, noise, epochs, timesteps)
    system.start()
