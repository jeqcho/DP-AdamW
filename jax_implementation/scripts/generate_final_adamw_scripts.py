import os

# --- Configuration ---
NUM_TRIALS = 5
CLIP_VALUE = 1.0
BASE_SEED = 1024
EXP_ENTITY = "cs2080"
EXP_GROUP = "jax_final_sweep"

# Define parameters per epsilon
epsilon_configs = {
    1.0: {
        "eps_vals": [5e-8, 1e-8],
        # "learning_rates": [0.00003, 0.00005, 0.00007, 0.0001, 0.0002],
        "learning_rates": [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002],
        "weight_decays": [0.01, 0.001, 0.0001, 0.00001] # Corrected based on user file list
    },
    3.0: {
        "eps_vals": [5e-8, 1e-8],
        "learning_rates": [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002],
        "weight_decays": [0.01, 0.001, 0.0001, 0.00001] # Corrected based on user file list
    },
    7.0: {
        "eps_vals": [5e-8, 1e-8],
        "learning_rates": [0.0007, 0.001, 0.0015, 0.002],
        "weight_decays": [0.01, 0.001, 0.0001, 0.00001]
    }
}

script_template = """#!/bin/bash
#SBATCH --job-name=jax_cifar_{epsilon}_wd{weight_decay}
#SBATCH --partition=kempner,kempner_h100
# #SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 1-23:00                                                          # Runtime in D-HH:MM
#SBATCH -o results/output_adamw_only_{epsilon}_wd{weight_decay}_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e results/output_adamw_only_{epsilon}_wd{weight_decay}_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --account kempner_sham_lab
# #SBATCH --account sneel_lab

CODE_DIR=/n/holylabs/LABS/sham_lab/Lab/lillian/DP-AdamW/jax_implementation
pushd $CODE_DIR

echo "Activating conda environment"
module load Mambaforge/23.11.0-fasrc01
conda activate dp-adamw
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cuda/12.4

# Fixed parameters for this script run
TARGET_EPS={epsilon}
DP_L2_NORM_CLIP={clip_value}
WEIGHT_DECAY={weight_decay}

# Wandb parameters
EXP_ENTITY="{exp_entity}"
EXP_GROUP="{exp_group}"
EXP_PROJ="cifar_${{TARGET_EPS}}_trials_2"

# Loop over parameters
eps_vals=({eps_vals_str})
learning_rates=({learning_rates_str})

for trial in {{1..{num_trials}}};
do
    echo "--- Starting Trial $trial ---"
    SEED=$(( {base_seed} + trial - 1 ))
    for e in "${{eps_vals[@]}}"
    do
        for lr in "${{learning_rates[@]}}"
        do
            echo "Running Trial $trial: lr=$lr, wd=$WEIGHT_DECAY, eps=$TARGET_EPS, clip=$DP_L2_NORM_CLIP, eps_val=$e, bias_corr=$bc_mode, seed=$SEED"

            weight_decay_arg="--weight_decay $WEIGHT_DECAY"

            # AdamW (without bias correction)
            trainer_args="--trainer DPAdamW"
            exp_suffix="AdamW"

            exp_name="${{TARGET_EPS}}_${{exp_suffix}}_wd${{WEIGHT_DECAY}}_lr${{lr}}_c${{DP_L2_NORM_CLIP}}_e${{e}}_trial${{trial}}"

            python main.py \\
                --seed $SEED \\
                --epochs 70 \\
                --progress_bar \\
                --batch_size 1024 \\
                --learning_rate $lr \\
                --dataset CIFAR10 \\
                --classifier_model CNN5 \\
                --activation tanh \\
                --dp_l2_norm_clip $DP_L2_NORM_CLIP \\
                $trainer_args \\
                $weight_decay_arg \\
                --eps_root $e \\
                --eps $e \\
                --target_eps $TARGET_EPS \\
                --exp_entity $EXP_ENTITY \\
                --exp_proj $EXP_PROJ \\
                --exp_group $EXP_GROUP \\
                --exp_name $exp_name

            echo "-----------------------------------------"
        done
    done
done

echo "DONE"
"""

# --- Script Generation ---
script_dir = os.path.dirname(__file__) or "."
count = 0

# Ensure results directory exists relative to script location
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

for epsilon, config in epsilon_configs.items():
    eps_vals = config["eps_vals"]
    eps_vals_str = " ".join(map(str, eps_vals))
    learning_rates = config["learning_rates"]
    weight_decays = config["weight_decays"]
    learning_rates_str = " ".join(map(str, learning_rates))

    for wd in weight_decays:
        filename = f"run_sweep_eps{epsilon}_wd{wd}_adamw_only.sh"
        filepath = os.path.join(script_dir, filename)

        content = script_template.format(
            epsilon=epsilon,
            weight_decay=wd,
            clip_value=CLIP_VALUE,
            eps_vals_str=eps_vals_str,
            exp_entity=EXP_ENTITY,
            exp_group=EXP_GROUP,
            learning_rates_str=learning_rates_str,
            num_trials=NUM_TRIALS,
            base_seed=BASE_SEED
        )

        with open(filepath, "w") as f:
            f.write(content)
        # Make executable
        os.chmod(filepath, 0o755)
        count += 1
        print(f"Generated: {filepath}")

print(f"\nGenerated {count} scripts.") 