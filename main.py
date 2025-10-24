# main.py

import os
import yaml
import itertools
import numpy as np
from tqdm import tqdm
import torch

from src.utils import set_seed
from src.model import FullyConnectedNet
from src.data_loader import get_mnist_data_loaders
from src.train import run_experiment

def main():
    # --- 1. Load Config ---
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Device Initialization ---
    print("\n--- Initializing Device ---")
    if config['device'] == 'cuda':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Successfully connected to GPU: {gpu_name}")
        else:
            print("WARNING: 'cuda' was specified, but no CUDA GPU found. Running on CPU.")
            config['device'] = 'cpu'
    else:
        print("Running on CPU as specified.")
    print("---------------------------\n")

    set_seed(config['seed'])
    
    # --- 3. Prepare Experiment Grid (Full List) ---
    all_pairs = list(itertools.combinations(range(10), 2))
    target_pairs = config['target_pairs'] if config['target_pairs'] is not None else all_pairs
    full_experiment_grid = []
    for pair in target_pairs:
        for error_type in config['error_types']:
            for error_rate in config['error_rates']:
                exp_id = f"{list(pair)[0]}v{list(pair)[1]}_{error_type}_{error_rate:.2f}"
                experiment = { 'id': exp_id, 'pair': list(pair), 'error_type': error_type, 'error_rate': error_rate, 'noise_source_class': None }
                if error_type == 'external_noise' and error_rate > 0:
                    possible_sources = [c for c in range(10) if c not in pair]
                    noise_source = np.random.choice(possible_sources)
                    experiment['noise_source_class'] = int(noise_source)
                full_experiment_grid.append(experiment)

    results_filepath = config['results_file']
    results_dir = os.path.dirname(results_filepath)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    completed_ids = set()
    all_results = []
    try:
        with open(results_filepath, 'r') as f:
            docs = yaml.safe_load_all(f)
            # 最初のドキュメント（config）は読み飛ばす
            next(docs, None) 
            # 2つ目以降のドキュメント（結果リスト）をロード
            for doc in docs:
                if doc:
                    all_results.extend(doc)

            for result in all_results:
                if 'experiment' in result and 'id' in result['experiment']:
                    completed_ids.add(result['experiment']['id'])
        if completed_ids:
             print(f"Found {len(completed_ids)} completed experiments in '{results_filepath}'. Resuming...")
    except (FileNotFoundError, StopIteration):
        print("No existing results file found or file is empty. Starting a new run.")
        # ファイルがなければ、configスナップショット付きで新規作成
        with open(results_filepath, 'w') as f:
            yaml.dump({'configuration_snapshot': config}, f, sort_keys=False, default_flow_style=False, indent=2)
            f.write("\n---\n")
    
    experiments_to_run = [exp for exp in full_experiment_grid if exp['id'] not in completed_ids]
    
    if not experiments_to_run:
        print("All experiments defined in the config are already complete. Nothing to do.")
        return

    pbar = tqdm(experiments_to_run, initial=len(completed_ids), total=len(full_experiment_grid))
    
    previous_pair = tuple(all_results[-1]['experiment']['pair']) if all_results else None
    previous_error_type = all_results[-1]['experiment']['error_type'] if all_results else None

    for exp_conditions in pbar:
        pbar.set_description(f"Running: {exp_conditions['id']}")
        current_id = exp_conditions.pop('id')
        
        train_loader, val_loader, test_loader = get_mnist_data_loaders(
            **exp_conditions,
            batch_size=config['training_params']['batch_size'],
            validation_split=config['training_params']['validation_split'],
            num_workers=config['training_params']['num_workers']
        )
        
        model = FullyConnectedNet(**config['model_params'])
        performance_metrics = run_experiment(model, train_loader, val_loader, test_loader, config)
        
        result_entry = {'experiment': {'id': current_id, **exp_conditions, **performance_metrics}}

        with open(results_filepath, 'a') as f:
            current_pair = tuple(exp_conditions['pair'])
            current_error_type = exp_conditions['error_type']
            if completed_ids or previous_pair is not None:
                if current_pair != previous_pair:
                    f.write("\n# --------------------------------------------------\n")
                    f.write(f"#   New Pair: {current_pair[0]} vs {current_pair[1]}\n")
                    f.write("# --------------------------------------------------\n\n")
                elif current_error_type != previous_error_type:
                    f.write("\n")
            
            yaml.dump([result_entry], f, sort_keys=False, default_flow_style=False)
            
            previous_pair = current_pair
            previous_error_type = current_error_type
            completed_ids.add(current_id)


    print(f"\nAll experiments completed. Results saved to {results_filepath}")

if __name__ == '__main__':
    main()