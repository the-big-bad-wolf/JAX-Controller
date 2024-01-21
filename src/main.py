import yaml
from Bathtub import Bathtub
from Classic import Classic
from CONSYS import CONSYS
import matplotlib.pyplot as plt

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
            learning_rate = float(params["learning_rate"])
            controller = Classic(0, [kp, ki, kd], learning_rate)
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

    plt.figure(1)
    plt.title("Learning Progression")
    plt.plot(system.MSE_history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.figure(2)
    plt.title("Control Parameters")
    plt.plot(system.param_history)
    plt.xlabel("Epoch")
    plt.ylabel("Parameter Value")
    plt.legend(["Kp", "Ki", "Kd"])
    plt.show()
