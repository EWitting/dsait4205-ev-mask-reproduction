from src.my_data_generation import *
from src.dvs_training import train_model
from src.dvs_testing import test_model
from pathlib import Path
import json

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Training parameters
TRAIN_CONFIGS = [
    {'epochs': 2, 'learning_rate': 0.001, 'layers': 'heads'},
    {'epochs': 5, 'learning_rate': 0.001, 'layers': 'heads'},
    [ # This run will use two stages
        {'epochs': 5, 'learning_rate': 0.001, 'layers': 'heads'},
        {'epochs': 10, 'learning_rate': 0.0001, 'layers': 'all'}
    ]
]
MULTIPLE_DIGITS = False
WINDOW_LENGTHS = [10,20,50]
WINDOW_SKIP = 50
TRAIN_DATA_PERCENTAGE = 0.8
SKIP_IF_EXISTS = True  # Skips retraining model if it already exists for the same combination of dataset and parameters
RESULTS_DIR = '../results'

if __name__ == '__main__':

    # Download Neuromorphic MNIST dataset
    train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
    test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    # Split into train and test
    split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=False, train_data_percentage=TRAIN_DATA_PERCENTAGE)

    # Generate RGB-D images and masks
    for window_len in WINDOW_LENGTHS:
        print(f'Generating RGB-D images and masks for {window_len}ms')
        path = f'../data/N_MNIST_images_{window_len}ms_skip_{WINDOW_SKIP}'
        generate_rgbd_images_and_masks(train_dataset, test_dataset, path, cleanup=False, window_len=window_len, skip=WINDOW_SKIP)

        print(f'Training models for {window_len}ms')
        for train_config in TRAIN_CONFIGS:
            model_path = train_model(path, train_config, multiple_digits=MULTIPLE_DIGITS, skip_if_exists=SKIP_IF_EXISTS)

            print('Evaluating model')
            results = test_model(model_path, path, multiple_digits=MULTIPLE_DIGITS, visualize_num=10)

            # Save results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            model_name = Path(model_path).stem
            with open(f'{RESULTS_DIR}/{model_name}.json', 'w') as f:
                results['window_len'] = window_len
                results['train_config'] = train_config
                results['multiple_digits'] = MULTIPLE_DIGITS
                results['model_path'] = model_path
                json.dump(results, f)
            print(f'Results saved to {RESULTS_DIR}/{model_name}.json')
