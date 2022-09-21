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
MODEL_NAME = "wandb_1353k0xc_model.pt"
CONFIG_FILENAME = "config_gnn.json"

MODEL_PATH = "/data/ltorres/CLOUD_CLASSIFICATION_WEIGHTS"
PIVOT_TENSORS_PATH = "pivot_nodes"


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
    
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME)))
    
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
    accuracy = utils.test_gnn_model(model, loader, pivot_tensors=samples, device=config["hardware"]["device"])
    
    print("Test accuracy: {:.2%}".format(accuracy))
    
    





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