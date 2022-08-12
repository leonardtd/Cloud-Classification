# Standard library imports
import argparse
import os
import sys
import pandas as pd

# Third party imports
import wandb 

# Local imports
from src.model import ModelTrainer
from src.utils import parse_arguments_train, set_seed, configure_model
from src import utils
from src import dataset

CONFIG_FILENAME = "config_gnn.json"
PROJECT_WANDB = "gcd-classification"
ENTITY = "leonardtd"

def is_there_arg(args, master_arg):
    if(master_arg in args):
        return True
    else:
        return False

def train(architecture, use_wandb, run_name, run_notes, num_experiments):
    set_seed(7)
    
    config_file = utils.get_config_filename(architecture)
    config = configure_model(config_file, use_wandb)

    ### read dataset
    print("Reading Dataset")
    
    train_paths = utils.get_gcd_paths(config["data"]["path_dataset"], "train")
    test_paths = utils.get_gcd_paths(config["data"]["path_dataset"], "test")
    
    train_dataset = dataset.GCDv2(train_paths, resize=config["data"]["resize"], use_augmentation=config["data"]["use_augmentation"])
    test_dataset = dataset.GCDv2(test_paths, resize=config["data"]["resize"], use_augmentation=False)
    
    print("train_dataset:", len(train_dataset))
    print("test_dataset:", len(test_dataset))
    print(50*"-")
    
    data_loaders = dict()
    data_loaders["train"] = utils.build_data_loader(train_dataset, config["hyperparameters"]["batch_size"], shuffle=True)
    data_loaders["test"] = utils.build_data_loader(test_dataset, config["hyperparameters"]["batch_size"], shuffle=True)
    
    results = pd.DataFrame()

    print(f"Starting {num_experiments} experiments")
    print(50*"-")
    for i in range(1, num_experiments+1):
        
        if use_wandb:
            wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=config, name=run_name, notes=run_notes, reinit=True)
            wandb.watch_called = False
            ## logger
            path_results = os.path.join(config["data"]["path_save_logs"], f"wandb_{wandb.run.id}_model.csv")
            
        else:
            path_results = os.path.join(config["data"]["path_save_logs"], f"results_model.csv")
        
        trainer = ModelTrainer(architecture, config, use_wandb, data_loaders)
        
        print(f"STARTING TRAINING: experiment {i}")
        model, data_logger = trainer.train()
        data_logger["experiment"] = i
        results = pd.concat([results, data_logger], axis=0)
        
        print(f"SAVING RESULTS: experiment {i}")
        wandb.finish()
    
    print(f"SAVING EXPERIMENT REULTS AT {path_results}")
    results.to_csv(path_results, header=True, index=False)



if __name__ == '__main__':
    use_sweep = is_there_arg(sys.argv, '--use_sweep')

    if not use_sweep:
        args = parse_arguments_train()
        use_wandb = args.wandb
        run_name = args.run_name
        run_notes = args.run_notes
        num_experiments = args.num_experiments
        architecture = args.architecture
    else:
        use_wandb = True
        run_name = None
        run_notes = None
        num_experiments = args.num_experiments
        architecture = args.architecture

    train(architecture, 
          use_wandb=use_wandb, 
          run_name=run_name, 
          run_notes=run_notes, 
          num_experiments=num_experiments)