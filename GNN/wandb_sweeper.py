# Import the W&B Python Library and log into W&B
from typing import Literal
import wandb
import os

from wandb.sdk.wandb_run import Run
from GNN import train
from GNN.configs import dpgcn

wandb.login()

project_id = "dp-gnn-sweeps2"


def override_and_run(
    run: Run,
    epsilon: float,
    optimizer: Literal["adamw", "adamwbc"],
    lr: float,
    weight_decay: float,
    run_name_suffix:str=""
):
    # Create workdir if it doesn't exist
    workdir = "./tmp"
    os.makedirs(workdir, exist_ok=True)

    # Load base config from dpgcn.py
    config = dpgcn.get_config()

    # Override epsilon and optimizer
    config.max_training_epsilon = epsilon
    config.optimizer = optimizer

    # Override specific parameters from sweep
    config.learning_rate = lr
    config.weight_decay = weight_decay

    # wandb overrides
    run.name = f"lr-{lr:.3e}-wd-{weight_decay:.3e}{run_name_suffix}"

    if optimizer == "adamw":
        config.eps = 1e-12
    else:
        config.eps = 2e-6

    # Save config to wandb
    wandb.config.update(dict(config))

    # Run training directly
    return train.train_and_evaluate(config, workdir)


def sweep_func_wrapper(epsilon: float, optimizer: Literal["adamw", "adamwbc"]):
    def sweep_func():
        # Initialize W&B run
        run = wandb.init(project=project_id, reinit=False)
        lr: float = wandb.config.lr
        weight_decay: float = wandb.config.weight_decay

        results = override_and_run(run, epsilon, optimizer, lr, weight_decay)
        
        return results

    return sweep_func


def sweep_once(epsilon: float, optimizer: Literal["adamw", "adamwbc"]):
    # 2: Define the search space
    sweep_configuration = {
        "name": f"sweep-{optimizer}-eps-{epsilon}",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "lr": {
                "max": 0.05,
                "min": 0.00001,
                "distribution": "log_uniform_values",
            },  # 1e-2 to 1e-3
            "weight_decay": {
                "max": 0.01,
                "min": 0.000001,
                "distribution": "log_uniform_values",
            },  # 1e-3 to 1e-6
        },
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_id)

    wandb.agent(sweep_id, function=sweep_func_wrapper(epsilon, optimizer), count=15)


if __name__ == "__main__":
    optimizers: list[Literal["adamw", "adamwbc"]] = ["adamwbc", "adamw"]
    for optimizer in optimizers:
        for epsilon in [3, 6, 12]:
            sweep_once(epsilon, optimizer)
