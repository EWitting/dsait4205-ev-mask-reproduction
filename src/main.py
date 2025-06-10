from src.my_data_generation import *
from src.dvs_training import train_model
from src.dvs_testing import test_model
from pathlib import Path
import json

from tensorflow.keras import backend as K
import gc
import torch

import os
import shutil

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
N_EXPERIMENT_RUNS = 4 # Number of times to rerun the experiment
SKIP_IF_EXISTS = False  # Skips retraining model if it already exists for the same combination of dataset and parameters
RESULTS_DIR = '../results'

DEPTH_CHANNEL_REPRESENTATION_MODE = 'window_time_normalized'  # 'original' (original from the paper) or 'window_time_normalized' (depth is calculated by normalizing the passed to time in the window) or 'zeros' (depth channel only has zero values)

def permanently_delete(path):
    """
    Permanently deletes a file or directory in a safe, cross-platform way.
    
    WARNING: This is irreversible and bypasses the system's recycle bin.
    """
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            print(f"Permanently deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Permanently deleted directory and all its contents: {path}")
        else:
            print(f"Error: Path not found: {path}")

    except FileNotFoundError:
        print(f"Error: Path not found: {path}")
    except PermissionError:
        print(f"Error: Permission denied to delete: {path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':

    if not os.path.exists('../data/NMINST'):
        # Download Neuromorphic MNIST dataset
        train_dataset = tonic.datasets.NMNIST(save_to='../data', train=True)
        test_dataset = tonic.datasets.NMNIST(save_to='../data', train=False)

    for i in range(N_EXPERIMENT_RUNS):

        if not os.path.exists('../data/N_MNIST') or N_EXPERIMENT_RUNS > 1:
            # Split into train and test, ensure that new split is made every run
            split_train_test_validation('../data/NMNIST', '../data/N_MNIST', cleanup=N_EXPERIMENT_RUNS > 1, train_data_percentage=TRAIN_DATA_PERCENTAGE)

        # Generate RGB-D images and masks
        for window_len in WINDOW_LENGTHS:
            print(f'Generating RGB-D images and masks for {window_len}ms')
            path = f'../data/N_MNIST_images_{window_len}ms_skip_{WINDOW_SKIP}'
            generate_rgbd_images_and_masks(train_dataset, test_dataset, path, cleanup=True, window_len=window_len, skip=WINDOW_SKIP, depth_channel_representation_mode=DEPTH_CHANNEL_REPRESENTATION_MODE)

            print(f'Training models for {window_len}ms')
            for train_config in TRAIN_CONFIGS:
                print("Clearing Keras session to release GPU memory...")
                K.clear_session()

                print("Running garbage collection...")
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("PyTorch CUDA cache cleared (for demonstration - no impact on TensorFlow).")
                
                print("Memory cleared.")

                model_path = train_model(path, train_config, depth_channel_representation_mode=DEPTH_CHANNEL_REPRESENTATION_MODE, multiple_digits=MULTIPLE_DIGITS, skip_if_exists=SKIP_IF_EXISTS, visualize_num=0)

                print('Evaluating model')
                results = test_model(model_path, path, multiple_digits=MULTIPLE_DIGITS, visualize_num=0, depth_channel_representation_mode=DEPTH_CHANNEL_REPRESENTATION_MODE)

                # Save results
                os.makedirs(RESULTS_DIR, exist_ok=True)
                model_name = Path(model_path).stem
                with open(f'{RESULTS_DIR}/{model_name}_run{i}.json', 'w') as f:
                    results['window_len'] = window_len
                    results['train_config'] = train_config
                    results['multiple_digits'] = MULTIPLE_DIGITS
                    results['model_path'] = model_path
                    json.dump(results, f)
                print(f'Results saved to {RESULTS_DIR}/{model_name}.json')
                
            # Clean up after processing each window length
            print(f"Cleaning up data folder: {path}")
            if os.path.exists(path):
                permanently_delete(path)
            
            # Clean up model folder
            model_dir = "../model"
            if os.path.exists(model_dir):
                print(f"Cleaning up model folder: {model_dir}")
                permanently_delete(model_dir)
                # Recreate empty directory
                os.makedirs(model_dir, exist_ok=True)
