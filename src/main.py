import jax.numpy as jnp
from jax.nn import sigmoid, relu
from jax.numpy import tanh
import jax
import numpy as np
import yaml
from Bathtub import Bathtub
from Rabbit import Rabbit
from Cournot import Cournot
from Classic import Classic
from AI import AI
from CONSYS import CONSYS
import matplotlib.pyplot as plt

with open("pivotal_parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
    plant = str(params["plant"])
    controller_name = str(params["controller"])
    noise = params["noise"]
    epochs = int(params["epochs"])
    timesteps = int(params["timesteps"])

    match controller_name:
        case "classic":
            classic_params = params["classic"]
            kp = float(classic_params["Kp"])
            ki = float(classic_params["Ki"])
            kd = float(classic_params["Kd"])
            learning_rate = float(params["learning_rate"])
            controller = Classic(0, jnp.array([kp, ki, kd]), learning_rate)
        case "AI":
            neural_net_params = params["neural_net"]
            nr_hidden_layers = int(neural_net_params["layers"])
            neurons_per_layer = int(neural_net_params["neurons_per_layer"])
            weight_bias_initial_range = neural_net_params["weight_bias_initial_range"]
            activation_function = str(neural_net_params["activation_function"])
            learning_rate = float(params["learning_rate"])

            layers = [3] + [neurons_per_layer for _ in range(nr_hidden_layers)] + [1]

            def gen_weights_bias(layers):
                sender = layers[0]
                weights_and_bias = []
                for receiever in layers[1:]:
                    weights = np.random.uniform(
                        weight_bias_initial_range[0],
                        weight_bias_initial_range[1],
                        (sender, receiever),
                    )
                    biases = np.random.uniform(
                        weight_bias_initial_range[0],
                        weight_bias_initial_range[1],
                        (1, receiever),
                    )
                    sender = receiever
                    weights_and_bias.append([weights, biases])
                return weights_and_bias

            match activation_function:
                case "sigmoid":
                    activation_function = sigmoid
                case "relu":
                    activation_function = relu
                case "tanh":
                    activation_function = tanh
                case _:
                    print("No valid activation function.")
                    exit()

            weights = gen_weights_bias(layers)

            controller = AI(
                0,
                weights,
                activation_function,
                learning_rate,
            )
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
        case "cournot":
            cournot_params = params["cournot"]
            p_max = float(cournot_params["p_max"])
            cm = float(cournot_params["cm"])
            q_1 = float(cournot_params["q_1"])
            q_2 = float(cournot_params["q_2"])
            target = float(cournot_params["profit_target"])
            plant = Cournot(p_max, cm, q_1, q_2)
        case "rabbit":
            rabbit_params = params["rabbit"]
            initial_rabbits = float(rabbit_params["initial_rabbits"])
            rabbit_birth_rate = float(rabbit_params["rabbit_birth_rate"])
            rabbit_eaten_rate = float(rabbit_params["rabbit_eaten_rate"])
            initial_wolves = float(rabbit_params["initial_wolves"])
            wolf_death_rate = float(rabbit_params["wolf_death_rate"])
            wolf_food_rate = float(rabbit_params["wolf_food_rate"])
            target = float(rabbit_params["target_rabbits"])
            plant = Rabbit(
                initial_rabbits,
                rabbit_birth_rate,
                rabbit_eaten_rate,
                initial_wolves,
                wolf_death_rate,
                wolf_food_rate,
            )
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

    if controller_name == "classic":
        plt.figure(2)
        plt.title("Control Parameters")
        plt.plot(system.param_history)
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Value")
        plt.legend(["Kp", "Ki", "Kd"])

    plt.show()
    file.close()
