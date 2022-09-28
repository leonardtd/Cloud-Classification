import argparse
import os
import sys
import pandas as pd
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src import utils
from src import dataset
from src.modules.graph_modules import GraphClassifier

###TEST_DIFFERENT MODELS
MODEL_NAME = "wandb_tact4jdo_model.pt"

CONFIG_FILENAME = "config_gnn.json"
PIVOT_TENSORS_PATH = "pivot_nodes"
SAVE_PATH = "/data/ltorres/predictions"

def load_model(config):
    model = GraphClassifier(
                     hidden_dim = config["model"]["hidden_dim"], 
                     num_hidden = config["model"]["num_hidden"], 
                     num_classes = config["model"]["num_classes"],
                     feature_extraction = config["model"]["feature_extraction"],
                     conv_type = config["model"]["conv_type"],
                     conv_parameters = config["model"]["conv_parameters"],
                     gnn_dropout = config["model"]["gnn_dropout"],
                     adjacency_builder = config["model"]["adjacency_builder"],
                     builder_parameter = config["model"]["builder_parameter"],
                     use_both_heads = config["model"]["use_both_heads"],
                )
    
    model.load_state_dict(torch.load(os.path.join(config['data']['path_save_weights'], MODEL_NAME)))
    
    return model.to(config["hardware"]["device"])

def predict(args):
    config = utils.parse_configuration(CONFIG_FILENAME)
        
    ### Read sampled tensors
    print("Loading pivot tensors")
    
    samples_path = os.path.join(PIVOT_TENSORS_PATH, f"{args.sampling_method}_sample.pt")
    samples = torch.load(samples_path)
    
    if args.use_centroids:
        centroids = torch.load(os.path.join(PIVOT_TENSORS_PATH, "centroids.pt"))
        centroids = torch.cat([t.clone().detach().view(1,-1).float() for t in list(centroids.values())], dim=0)
        
        samples = torch.cat((samples,centroids), dim=0)
        
    print("Number of pivot samples: {}".format(samples.shape[0]))
    print(50*"-")
    
    ### Load Model
    print("Loading Model")
    print(50*"-")
    model = load_model(config)
    
    ### Dataloader
    print("Reading inference data")
    print(50*"-")
    test_paths = utils.get_gcd_paths(config["data"]["path_dataset"], "test")
    test_dataset = dataset.GCD(test_paths, resize=config["data"]["resize"], use_augmentation=False)
    
    print("test_dataset:", len(test_dataset))
    print(50*"-")
    
    loader = utils.build_data_loader(test_dataset, batch_size=1, shuffle=True)
    
    ### Prediction
    results, accuracy = utils.predict_gnn_model(model, loader, pivot_tensors=samples, device=config["hardware"]["device"])

    print("Test accuracy: {:.2%}".format(accuracy))
    print("SAVING RESULTS")
    results.to_csv(os.path.join(SAVE_PATH, f"predictions_{args.sampling_method}.csv"), index=False)
    


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-s', '--sampling_method', default='stratified',
                    help="Sampling method for pivot nodes")
    
    ap.add_argument('-c', '--use_centroids', action='store_true', default=False,
                    help="Use class centroids as pivot nodes")
    ap.add_argument('-n  ', '--no-use_centroids', dest='use_centroids', action='store_false',
                    help=" Do not use class centroids as pivot nodes")
    
    args = ap.parse_args()
    
    
    predict(args)
    
    
"""
{
    "hardware": {
        "device": "cuda:3"
    },
    "data": {
        "path_dataset": "/data/ltorres",
        "path_save_weights": "/data/ltorres/CLOUD_CLASSIFICATION_WEIGHTS",
        "path_save_logs": "/data/ltorres/model_logs",
        "class_names": ["1_cumulus", "2_altocumulus", "3_cirrus", "4_clearsky", "5_stratocumulus", "6_cumulonimbus", "7_mixed"],
        "resize": 256,
        "use_augmentation": true
    },
    "hyperparameters": {
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.00005,
        "early_stopping_tolerance": 8,
        "criterion": "cross_entropy",
        "optimizer": "sgd",
        "use_scheduler": true,
        "lr_decay_steps": 15,
        "lr_decay_gamma": 0.5
    },
    "model": {
        "hidden_dim": 512,
        "num_hidden": 2,
        "num_classes": 7,
        "feature_extraction": false,
        "conv_type": "gcn",
        "conv_parameters": {"num_heads":2, "agg":"sum"},
        "gnn_dropout": 0.3,
        "adjacency_builder": "pearson_corr",
        "builder_parameter": 0.7,
        "use_both_heads": false,
        "loss_lambda": 1
    }
}
"""