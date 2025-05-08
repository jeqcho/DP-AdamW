# Import the W&B Python Library and log into W&B
import wandb
import os
from GNN import train
from GNN.configs import dpgcn

wandb.login()

project_id = "dp-gnn-sweeps2"


def sweep_func():
    # Initialize W&B run
    run = wandb.init(project=project_id, reinit=False)
    lr: float = wandb.config.lr
    weight_decay: float = wandb.config.weight_decay
    
    # Create workdir if it doesn't exist
    workdir = "./tmp"
    os.makedirs(workdir, exist_ok=True)
    
    # Load base config from dpgcn.py
    config = dpgcn.get_config()
    
    # Override specific parameters
    config.learning_rate = lr
    config.weight_decay = weight_decay
    config.wandb_project = project_id
    config.experiment_name = run.name
    config.group = run.group if run.group else 'sweep'
    
    # Save config to wandb
    wandb.config.update(dict(config))
    
    # Run training directly
    train.train_and_evaluate(config, workdir)


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "lr": {"max": 0.02, "min": 0.005},  # 1e-2 to 1e-3
        "weight_decay": {"max": 0.001, "min": 0.000001},  # 1e-3 to 1e-6
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_id)

wandb.agent(sweep_id, function=sweep_func, count=10)
