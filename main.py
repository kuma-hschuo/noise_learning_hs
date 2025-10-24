
import yaml
import itertools
import numpy as np
from tqdm import tqdm
import time

from src.utils import set_seed
from src.model import FullyConnectedNet
from src.data_loader import get_mnist_data_loaders
from src.train import run_experiment

def main():
    # --- 1. Load Config ---
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Set Seed for Reproducibility ---
    set_seed(config['seed'])

    # --- 3. Prepare Experiment Grid ---
    all_pairs = list(itertools.combinations(range(10), 2))
    target_pairs = config['target_pairs'] if config['target_pairs'] is not None else all_pairs
    
    experiment_grid = []
    for pair in target_pairs:
        for error_type in config['error_types']:
            for error_rate in config['error_rates']:
                experiment = {
                    'pair': list(pair),
                    'error_type': error_type,
                    'error_rate': error_rate,
                    'noise_source_class': None
                }
                if error_type == 'external_noise' and error_rate > 0:
                    # Choose a noise source class that is not in the pair
                    possible_sources = [c for c in range(10) if c not in pair]
                    noise_source = np.random.choice(possible_sources)
                    experiment['noise_source_class'] = int(noise_source) # Ensure YAML serializable
                experiment_grid.append(experiment)

    # --- 4. Run All Experiments ---
    all_results = []
    pbar = tqdm(experiment_grid)
    for i, exp_conditions in enumerate(pbar):
        exp_id = f"{exp_conditions['pair'][0]}v{exp_conditions['pair'][1]}_{exp_conditions['error_type']}_{exp_conditions['error_rate']:.2f}"
        pbar.set_description(f"Running: {exp_id}")
        
        # --- Data Loading ---
        train_loader, val_loader, test_loader = get_mnist_data_loaders(
            pair=exp_conditions['pair'],
            error_type=exp_conditions['error_type'],
            error_rate=exp_conditions['error_rate'],
            noise_source_class=exp_conditions['noise_source_class'],
            batch_size=config['training_params']['batch_size'],
            validation_split=config['training_params']['validation_split']
        )
        
        # --- Model Initialization ---
        model = FullyConnectedNet(**config['model_params'])
        
        # --- Run Training and Evaluation ---
        performance_metrics = run_experiment(model, train_loader, val_loader, test_loader, config)
        
        # --- Store Results ---
        result_entry = {
            'experiment': {
                'id': exp_id,
                **exp_conditions,
                **performance_metrics
            }
        }
        all_results.append(result_entry)

        # --- Save results incrementally ---
        with open(config['results_file'], 'w') as f:
            yaml.dump(all_results, f, sort_keys=False, default_flow_style=False)

    print(f"\nAll experiments completed. Results saved to {config['results_file']}")

if __name__ == '__main__':
    main()
