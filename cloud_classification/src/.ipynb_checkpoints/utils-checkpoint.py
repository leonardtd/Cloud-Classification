import os
import argparse
import random
import json
import glob
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from tqdm import tqdm

from sklearn.metrics import accuracy_score

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
    ap.add_argument('-x', '--num_experiments', default=1, type=int,
                    help="number of experiments to run")

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
#     config_wandb = dict(
#         device = config_file["hardware"]["device"],
        
#         # data
#         resize = config_file["data"]["resize"],
#         use_augmentation = config_file["data"]["use_augmentation"],
        
#         # hyperparameters
#         epochs = config_file["hyperparameters"]["epochs"],
#         batch_size = config_file["hyperparameters"]["batch_size"],
#         learning_rate = config_file["hyperparameters"]["learning_rate"],
#         early_stopping_tolerance = config_file["hyperparameters"]["early_stopping_tolerance"],
#         criterion = config_file["hyperparameters"]["criterion"],
#         optimizer = config_file["hyperparameters"]["optimizer"],
#         use_scheduler = config_file["hyperparameters"]["use_scheduler"],
#         lr_decay_steps = config_file["hyperparameters"]["lr_decay_steps"],
#         lr_decay_gamma = config_file["hyperparameters"]["lr_decay_gamma"],
        
#         # model
#         hidden_dim = config_file["model"]["hidden_dim"],
#         num_hidden = config_file["model"]["num_hidden"],
#         num_classes = config_file["model"]["num_classes"],
#         conv_type = config_file["model"]["conv_type"],
#         conv_parameters = config_file["model"]["conv_parameters"],
#         gnn_dropout = config_file["model"]["gnn_dropout"],
#         adjacency_builder = config_file["model"]["adjacency_builder"],
#         builder_parameter = config_file["model"]["builder_parameter"],
#         use_both_heads = config_file["model"]["use_both_heads"],
#         loss_lambda = config_file["model"]["loss_lambda"],
#     )

    return config_file


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
        num_workers=6,
        shuffle=shuffle,
    )
    
    return data_loader


def loge_loss(x , labels):
    
    epsilon = 1 - math.log(2)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(x, labels)
   
    loss = torch.mean(torch.log(epsilon + loss) - math.log(epsilon))
    
    return loss
    
    
def build_criterions(config):
    
    criterions = {}
    criterion_name = config["hyperparameters"]["criterion"]
    
    if criterion_name == 'cross_entropy':
        main_criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'loge':
        main_criterion = loge_loss
    else:
        raise NotImplementedError(f"{criterion_name} is not a valid criterion")
        
    criterions["main_head"] = main_criterion
    
    if config["model"]["use_both_heads"]:
        criterions["second_head"] = nn.CrossEntropyLoss()
    
    return criterions
        
    
def build_optimizer(model, config):
    
    optim_name = config["hyperparameters"]["optimizer"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    
    if optim_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    elif optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=learning_rate)  
    else:
        raise NotImplementedError(f"{optim_name} is not a valid optimizer")
        
def build_scheduler(optimizer, config):
    
    lr_decay_steps = config["hyperparameters"]["lr_decay_steps"]
    lr_decay_gamma = config["hyperparameters"]["lr_decay_gamma"]
    
    return lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_gamma)

def get_matrix_density(tensor):
    with torch.no_grad():
        density = tensor.sum().item()/(tensor.flatten().shape[0])
    return density


####################################
# 2. TRAINING
####################################

def train_model(model, data_loader, criterions, optimizer, device, use_both_heads, loss_lambda):
    model.train()
    
    fin_loss = 0
    fin_density = 0
    fin_preds = []
    fin_targs = []

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
                

        optimizer.zero_grad()
        
        logits_main_head, logits_second_head, density = model(data["images"])
        fin_density += density
        
        if use_both_heads:
            loss = criterions["main_head"](logits_main_head, data["targets"]) + loss_lambda*criterions["second_head"](logits_second_head, data["targets"])
        else:
            loss = criterions["main_head"](logits_main_head, data["targets"])
        
        loss.backward()
        
        optimizer.step()
        
        fin_loss += loss.item()

        batch_preds = F.softmax(logits_main_head, dim=1)
        batch_preds = torch.argmax(batch_preds, dim=1)

        fin_preds.append(batch_preds.cpu().numpy())
        fin_targs.append(data["targets"].cpu().numpy())
    
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    accuracy = accuracy_score(targets, predictions)
    loss = fin_loss / len(data_loader)
    density = fin_density / len(data_loader)
    
    return loss, accuracy, targets, predictions, density


def test_model(model, data_loader, criterions, device, use_both_heads, loss_lambda):
    model.eval()
    
    fin_loss = 0
    fin_density = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            logits_main_head, logits_second_head, density = model(data["images"])
            fin_density += density
            
            if use_both_heads:
                loss = criterions["main_head"](logits_main_head, data["targets"]) + loss_lambda*criterions["second_head"](logits_second_head, data["targets"])
            else:
                loss = criterions["main_head"](logits_main_head, data["targets"])
            
            fin_loss += loss.item()

            batch_preds = F.softmax(logits_main_head, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())
            
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    accuracy = accuracy_score(targets, predictions)
    loss = fin_loss / len(data_loader)
    density = fin_density / len(data_loader)
    
    return loss, accuracy, targets, predictions, density