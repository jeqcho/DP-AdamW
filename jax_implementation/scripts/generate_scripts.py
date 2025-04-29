import os

# Define parameter ranges
clips = [1.0, 2.0, 3.0]
eps_vals_str = ['5e-8', '1e-8'] # String representation for filename and arg
epsilons = [1.0, 3.0, 7.0]
learning_rates = [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
weight_decays = ['Adam', 0.1, 0.01, 0.001, 0.0005, 0.0001]

# Ensure results directory exists if scripts are run from here
# Note: SLURM output files go to 'results/' relative to where sbatch is called
# If sbatch is called from the scripts directory, 'results/' needs to exist there.
# Consider adjusting output paths in the template if needed.
# os.makedirs("results", exist_ok=True) # Commented out as results dir might be relative to workspace root

script_template = """#!/bin/bash
#SBATCH --job-name=jax_cifar_{epsilon}_c{clip}_e{eps_val_str}
#SBATCH --partition=kempner,kempner_h100
# #SBATCH --partition=seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 1-23:00                                                          # Runtime in D-HH:MM
#SBATCH -o results/output_{epsilon}_c{clip}_e{eps_val_str}_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e results/output_{epsilon}_c{clip}_e{eps_val_str}_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --account kempner_sham_lab
# #SBATCH --account sneel_lab


echo "Activating conda environment"
module load Mambaforge/23.11.0-fasrc01
conda activate dp-adamw
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cuda/12.4

# Fixed parameters for this script
TARGET_EPS={epsilon}
DP_L2_NORM_CLIP={clip}
EPS_VAL={eps_val_str} # This value is used for both --eps and --eps_root

# Wandb parameters
EXP_ENTITY="cs2080"
EXP_GROUP="jax"
EXP_PROJ="cifar_${{TARGET_EPS}}"

# Loop over learning rates and weight decays
learning_rates=({learning_rates_str})
weight_decays=({weight_decays_str})

for lr in "${{learning_rates[@]}}"
do
    for wd in "${{weight_decays[@]}}"
    do
        echo "Running with lr=$lr, wd=$wd, eps=$TARGET_EPS, clip=$DP_L2_NORM_CLIP, eps_val=$EPS_VAL"

        # Construct experiment name and command arguments based on weight decay
        if [ "$wd" == "Adam" ]; then
            # Adam (no weight decay)
            exp_name="${{TARGET_EPS}}_Adam_${{lr}}_${{DP_L2_NORM_CLIP}}_${{EPS_VAL}}"
            weight_decay_arg=""
            trainer_args="--trainer DPAdam --adam_corr"
        else
            # AdamW (with weight decay)
            exp_name="${{TARGET_EPS}}_AdamW_${{wd}}_${{lr}}_${{DP_L2_NORM_CLIP}}_${{EPS_VAL}}"
            weight_decay_arg="--weight_decay $wd"
            trainer_args="--trainer DPAdamW --adam_corr" # Use --adam_corr for AdamW
        fi

        python ../main.py \\
            --seed 1024 \\
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
            --eps_root $EPS_VAL \\
            --eps $EPS_VAL \\
            --target_eps $TARGET_EPS \\
            --exp_entity $EXP_ENTITY \\
            --exp_proj $EXP_PROJ \\
            --exp_group $EXP_GROUP \\
            --exp_name $exp_name

        echo "-----------------------------------------"
    done
done

echo "DONE"
"""

learning_rates_str = " ".join(map(str, learning_rates))
weight_decays_str = " ".join(map(str, weight_decays))

count = 0
# Ensure we are generating files in the same directory as the script
script_dir = os.path.dirname(__file__)
if not script_dir:
    script_dir = "." # Handle case where script is run from its own directory

# Create results directory relative to script directory if it doesn't exist
# Adjust this if SBATCH output should go elsewhere relative to the sbatch call location
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)


for epsilon in epsilons:
    for clip in clips:
        for eps_val_str in eps_vals_str:

            filename = f"run_eps{epsilon}_clip{clip}_eps{eps_val_str}.sh"
            filepath = os.path.join(script_dir, filename)

            content = script_template.format(
                epsilon=epsilon,
                clip=clip,
                eps_val_str=eps_val_str,
                learning_rates_str=learning_rates_str,
                weight_decays_str=weight_decays_str
            )

            with open(filepath, "w") as f:
                f.write(content)
            # Make the script executable
            os.chmod(filepath, 0o755)
            count += 1
            print(f"Generated {filepath}")

print(f"\nGenerated {count} scripts.") 