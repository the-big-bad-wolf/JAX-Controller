import yaml

with open("pivotal_parameters.yaml", "r") as file:
    data = yaml.safe_load(file)
    bathtub_params = data["bathtub"]
    A = float(bathtub_params["A"])
    C = float(bathtub_params["C"])
    H0 = float(bathtub_params["H0"])
    print(A, C, H0)
