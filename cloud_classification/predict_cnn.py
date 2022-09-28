import argparse
import os
import sys
import pandas as pd
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src import utils
from src import dataset
from src.modules.conv_modules import CloudNet, ResNet50Classifier

###TEST_DIFFERENT MODELS
MODEL_NAME = "wandb_2iryxig7_model.pt"

CONFIG_FILENAME = "config_cnn.json"
SAVE_PATH = "/data/ltorres/predictions"

def load_model(config):
    if config['model']['type'] == "cloudnet":
        model = CloudNet(
                out_dims = config["model"]["num_classes"],
                dropout = config["model"]["dropout"]
            )
    elif config['model']['type'] == "resnet":
        model = ResNet50Classifier(config["model"]["num_classes"], feature_extraction=False)
    
    model.load_state_dict(torch.load(os.path.join(config['data']['path_save_weights'], MODEL_NAME)))
    
    return model.to(config["hardware"]["device"])

def predict():
    config = utils.parse_configuration(CONFIG_FILENAME)
        
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
    
    loader = utils.build_data_loader(test_dataset, batch_size=40, shuffle=False)
    
    ### Prediction
    results, accuracy = utils.predict_cnn_model(model, loader, device=config["hardware"]["device"])

    print("Test accuracy: {:.2%}".format(accuracy))
    print("SAVING RESULTS")
    results.to_csv(os.path.join(SAVE_PATH, f"predictions_{config['model']['type']}.csv"), index=False)
    


if __name__ == '__main__':
    predict()
    