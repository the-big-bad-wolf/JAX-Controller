import jax.numpy as jnp
from jax.nn import sigmoid, relu
from jax.numpy import tanh
import jax
import yaml
from Bathtub import Bathtub
from Cournot import Cournot
from Classic import Classic
from AI import AI
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
            controller = Classic(0, jnp.array([kp, ki, kd]), learning_rate)
        case "AI":
            neural_net_params = params["neural_net"]
            layers = int(neural_net_params["layers"])
            neurons_per_layer = int(neural_net_params["neurons_per_layer"])
            weight_bias_initial_range = neural_net_params["weight_bias_initial_range"]
            activation_function = str(neural_net_params["activation_function"])
            learning_rate = float(params["learning_rate"])

            # Input layer
            weights = jax.random.uniform(
                jax.random.PRNGKey(0),
                shape=(1, 3, 3),
                minval=weight_bias_initial_range[0],
                maxval=weight_bias_initial_range[1],
            )
            # Hidden layers
            weights = jnp.append(
                weights,
                jax.random.uniform(
                    jax.random.PRNGKey(1),
                    shape=(1, neurons_per_layer, 3),
                    minval=weight_bias_initial_range[0],
                    maxval=weight_bias_initial_range[1],
                ),
            )
            weights = jnp.append(
                weights,
                jax.random.uniform(
                    jax.random.PRNGKey(2),
                    shape=(layers - 1, neurons_per_layer, neurons_per_layer),
                    minval=weight_bias_initial_range[0],
                    maxval=weight_bias_initial_range[1],
                ),
            )
            # Output layer
            weights = jnp.append(
                weights,
                jax.random.uniform(
                    jax.random.PRNGKey(3),
                    shape=(1, 1, neurons_per_layer),
                    minval=weight_bias_initial_range[0],
                    maxval=weight_bias_initial_range[1],
                ),
            )
            # Input layer
            biases = jax.random.uniform(
                jax.random.PRNGKey(4),
                shape=(1, 3),
                minval=weight_bias_initial_range[0],
                maxval=weight_bias_initial_range[1],
            )
            # Hidden layers
            biases = jnp.append(
                biases,
                jax.random.uniform(
                    jax.random.PRNGKey(5),
                    shape=(layers, neurons_per_layer),
                    minval=weight_bias_initial_range[0],
                    maxval=weight_bias_initial_range[1],
                ),
            )
            # Output layer
            biases = jnp.append(
                biases,
                jax.random.uniform(
                    jax.random.PRNGKey(6),
                    shape=(1, 1),
                    minval=weight_bias_initial_range[0],
                    maxval=weight_bias_initial_range[1],
                ),
            )

            print("Weights: ", weights)
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

            controller = AI(
                0,
                weights,
                biases,
                layers,
                neurons_per_layer,
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

    file.close()
