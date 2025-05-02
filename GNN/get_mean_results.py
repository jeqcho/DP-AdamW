import pandas as pd
import numpy as np
from GNN.wandb_sweeper import override_and_run
import wandb

if __name__ == "__main__":
    project_id = "dp-gnn-sweeps2"
    
    # Read final_params.csv
    df = pd.read_csv("GNN/results/final_params.csv")
    
    # Initialize wandb
    wandb.login()
    
    # Create a list to store results
    results = []
    
    # For each row in the dataframe
    for _, row in df.iterrows():
        if pd.isna(row['test_accuracy']):  # Skip rows with missing test accuracy
            continue
            
        # Run 5 times for each configuration
        accuracies = []
        print(f"Running {row['epsilon']=} {row['optimizer']=}")
        for i in range(5):
            run = wandb.init(project=project_id)
            result = override_and_run(
                run=run,
                epsilon=row['epsilon'],
                optimizer=row['optimizer'],
                lr=row['lr'],
                weight_decay=row['weight_decay'],
                run_name_suffix=f"{row['epsilon']}_{row['optimizer']}_{i}"
            )
            accuracies.append(float(result))
            run.finish()
        
        # Calculate mean and std of accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # Store results
        results.append({
            'optimizer': row['optimizer'],
            'epsilon': row['epsilon'],
            'lr': row['lr'],
            'weight_decay': row['weight_decay'],
            'mean_test_accuracy': mean_acc,
            'std_test_accuracy': std_acc,
            'original_test_accuracy': row['test_accuracy']
        })
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("GNN/results/mean_results.csv", index=False)