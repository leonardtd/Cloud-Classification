import os
import argparse
import random
import json
import glob

import numpy as np
import torch


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def parse_arguments_train():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-n  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-a', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-t', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

    args = ap.parse_args()

    return args


def parse_configuration(config_file):
    """
    Loads config file if a string was passed
    and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file
    
def configure_model(config_file, use_wandb):
    config_file = parse_configuration(config_file)
    config = dict(
        device = config_file["hardware"]["device"],
        
        # data
        path_dataset = config_file["data"]["path_dataset"],
        path_save_weights = config_file["data"]["path_save_weights"],
        resize = config_file["data"]["resize"],
        use_augmentation = config_file["data"]["use_augmentation"],
        
        # hyperparameters
        epochs = config_file["hyperparameters"]["epochs"],
        batch_size = config_file["hyperparameters"]["batch_size"],
        learning_rate = config_file["hyperparameters"]["learning_rate"],
        lr_decay_steps = config_file["hyperparameters"]["lr_decay_steps"],
        lr_decay_gamma = config_file["hyperparameters"]["lr_decay_gamma"],
        early_stopping_tolerance = config_file["hyperparameters"]["early_stopping_tolerance"],
        loss_fn = config_file["hyperparameters"]["loss_fn"],
        optimizer = config_file["hyperparameters"]["optimizer"],
        
        # model
        hidden_dim = config_file["model"]["hidden_dim"],
        num_hidden = config_file["model"]["num_hidden"],
        num_classes = config_file["model"]["num_classes"],
        conv_type = config_file["model"]["conv_type"],
        conv_parameters = config_file["model"]["conv_parameters"],
        graph_builder = config_file["model"]["graph_builder"],
        builder_parameter = config_file["model"]["builder_parameter"],
        use_both_heads = config_file["model"]["use_both_heads"],
    )

    if not use_wandb:
        config = type("configuration", (object,), config)

    return config


####################################
# 1. FUNCTIONS
####################################

## Dataset
def get_gcd_paths(data_dir, dataset_type):
    return glob.glob(
                     os.path.join(data_dir,f'GCD/{dataset_type}/**/*.jpg'), 
                     recursive=True
                    )


def get_gcd_targets(paths):
    targets = np.array(list(map(int,[os.path.basename(x).split('_')[0] 
                                    for x in paths])))
    targets -= 1 # starts from 0
    
    return targets


def build_data_loader(dataset, batch_size, shuffle=False):
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=-1,
        shuffle=shuffle,
    )
    
    return data_loader
    