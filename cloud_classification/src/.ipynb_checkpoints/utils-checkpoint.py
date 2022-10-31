import os
import argparse
import random
import json
import glob
import math

import numpy as np
import pandas as pd
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
    ap.add_argument('-r', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-t', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")
    ap.add_argument('-x', '--num_experiments', default=1, type=int,
                    help="number of experiments to run")
    ap.add_argument('-a', '--architecture', required=False, type=str, default="gnn",
                    help="name of the model architecture")

    args = ap.parse_args()

    return args

def get_config_filename(architecture:str):
    return f"config_{architecture}.json"

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
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    loss = criterion(x, labels)
   
    loss = torch.mean(torch.log(epsilon + loss) - math.log(epsilon))
    
    return loss
    
    
def build_criterions(architecture, config):
    
    criterions = {}
    criterion_name = config["hyperparameters"]["criterion"]
    
    if criterion_name == 'cross_entropy':
        main_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif criterion_name == 'loge':
        main_criterion = loge_loss
    else:
        raise NotImplementedError(f"{criterion_name} is not a valid criterion")
        
    criterions["main_head"] = main_criterion
    
    if architecture == "gnn":
        if config["model"]["use_both_heads"]:
            criterions["second_head"] = nn.CrossEntropyLoss()
    
    return criterions
        
    
def build_optimizer(model, config):
    
    optim_name = config["hyperparameters"]["optimizer"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    
    if optim_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-05)
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


def one_hot_encoding(num_samples, num_classes, index, labels):
    onehot = torch.zeros((num_samples, num_classes))
    onehot[index,labels[index]] = 1
    
    return onehot

####################################
# 2. TRAINING
####################################

def train_gnn_model(model, data_loader, criterions, optimizer, device, use_both_heads, loss_lambda, num_classes=7):
    model.train()
    
    fin_loss = 0
    fin_density = 0
    fin_preds = []
    fin_targs = []

    for data in tqdm(data_loader, total=len(data_loader)):
        
        ### Label prop
        num_nodes = data["targets"].shape[0]
        
        random_permutation = torch.randperm(num_nodes)
        
        Dl_train_idx = random_permutation[:num_nodes//2] # use these nodes to label propagate
        Du_train_idx = random_permutation[num_nodes//2:] # use these to train (backpropagation)

        one_hot_labels = one_hot_encoding(num_nodes, num_classes, Dl_train_idx, data["targets"]).to(device)
        
        for k, v in data.items():
            data[k] = v.to(device)
                

        optimizer.zero_grad()
        
        logits_main_head, logits_second_head, density = model(data["images"], one_hot_labels)
        fin_density += density
        
        loss = criterions["main_head"](logits_main_head[Du_train_idx], data["targets"][Du_train_idx])
        
        fin_loss += loss.item() ### ONLY LOG MAIN HEAD LOSS
        
        if use_both_heads:
            loss = loss + loss_lambda*criterions["second_head"](logits_second_head, data["targets"])

        loss.backward()
        
        optimizer.step()

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


def test_gnn_model(model, data_loader, criterions, device, use_both_heads, loss_lambda, num_classes):
    model.eval()
    
    fin_loss = 0
    fin_density = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            
            ### Label prop
            num_nodes = data["targets"].shape[0]

            random_permutation = torch.randperm(num_nodes)

            Dl_train_idx = random_permutation[:num_nodes//2] # use these nodes to label propagate
            Du_train_idx = random_permutation[num_nodes//2:] # use these to train (backpropagation)

            one_hot_labels = one_hot_encoding(num_nodes, num_classes, Dl_train_idx, data["targets"]).to(device)
            
            for k, v in data.items():
                data[k] = v.to(device)

            logits_main_head, logits_second_head, density = model(data["images"], one_hot_labels)
            fin_density += density
            
            loss = criterions["main_head"](logits_main_head[Du_train_idx], data["targets"][Du_train_idx])
            fin_loss += loss.item()
            
            if use_both_heads:
                loss = loss + loss_lambda*criterions["second_head"](logits_second_head, data["targets"])
            
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


def train_cnn_model(model, data_loader, criterions, optimizer, device):
    model.train()
    
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
                

        optimizer.zero_grad()
        
        logits = model(data["images"])
        loss = criterions["main_head"](logits, data["targets"])
        loss.backward()
        
        optimizer.step()
        
        fin_loss += loss.item()

        batch_preds = F.softmax(logits, dim=1)
        batch_preds = torch.argmax(batch_preds, dim=1)

        fin_preds.append(batch_preds.cpu().numpy())
        fin_targs.append(data["targets"].cpu().numpy())
    
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    accuracy = accuracy_score(targets, predictions)
    loss = fin_loss / len(data_loader)
    
    return loss, accuracy, targets, predictions


def test_cnn_model(model, data_loader, criterions, device):
    model.eval()
    
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            logits = model(data["images"])
            loss = criterions["main_head"](logits, data["targets"])
            
            fin_loss += loss.item()

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())
            
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    accuracy = accuracy_score(targets, predictions)
    loss = fin_loss / len(data_loader)
    
    return loss, accuracy, targets, predictions


########### PREDICTION UTILS

def predict_gnn_model(model, encoded_sample_labels, data_loader, pivot_tensors, device):
    model.eval()
    
    fin_loss = 0
    fin_density = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            image = data["images"].to(device)
            single_logits = model.get_deep_features(image)
            
            input = torch.cat((single_logits, pivot_tensors.to(device)), dim=0)
            encoded_labels = torch.cat([torch.zeros(1,7), encoded_sample_labels],dim=0).to(device)
            
            logits = model.predict(input, encoded_labels)[0].unsqueeze(0) ### ONLY INTERESTED IN THE TEST SAMPLE PREDICTION
            
            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())
            
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    results = pd.DataFrame({"targets": targets, "predictions": predictions})
    accuracy = accuracy_score(targets, predictions)

    return results, accuracy


def predict_cnn_model(model, data_loader, device):
    model.eval()
    
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            logits = model(data["images"])

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())
            
    targets = np.concatenate(fin_targs, axis=0)
    predictions = np.concatenate(fin_preds, axis=0)
    
    results = pd.DataFrame({"targets": targets, "predictions": predictions})
    accuracy = accuracy_score(targets, predictions)

    return results, accuracy