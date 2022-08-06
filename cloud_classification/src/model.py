import os
import pandas as pd
import torch
import wandb

from . import utils
from .modules.graph_modules import GraphClassifier


def build_model(config):
    model = GraphClassifier(
                 hidden_dim = config["model"]["hidden_dim"], 
                 num_hidden = config["model"]["num_hidden"], 
                 num_classes = config["model"]["num_classes"],
                 conv_type = config["model"]["conv_type"],
                 conv_parameters = config["model"]["conv_parameters"],
                 adjacency_builder = config["model"]["adjacency_builder"],
                 builder_parameter = config["model"]["builder_parameter"],
                 use_both_heads = config["model"]["use_both_heads"],
            )
    
    return model.to(config["hardware"]["device"])
    

class ModelTrainer:
    def __init__(self, config, use_wandb, data_loaders:dict):
        
        self.use_wandb = use_wandb
        self.config = config
        self.data_loaders = data_loaders
        
        self.model = build_model(self.config)
        
        self.criterions = utils.build_criterions(self.config)
        self.optimizer = utils.build_optimizer(self.model, self.config)
        self.scheduler = utils.build_scheduler(self.optimizer, self.config) if config["hyperparameters"]["use_scheduler"] else None
        
        # logger
        self.data_logger = pd.DataFrame(columns=["type", "epoch", "loss", "accuracy"])
        
        if self.use_wandb:
            wandb.watch(self.model, log="all")
            self.path_artifact = os.path.join(self.config["data"]["path_save_weights"], f"wandb_{wandb.run.id}_model.pt")
        else:
            self.path_artifact = os.path.join(self.config["data"]["path_save_weights"], f"parameters_model.pt")
            
        ### training parameters
        self.best_accuracy = 0
        self.early_stopping_tolerance = self.config["hyperparameters"]["early_stopping_tolerance"]
            
    def train(self):
        
        epochs = self.config["hyperparameters"]["epochs"]
        device = self.config["hardware"]["device"]
        use_both_heads = self.config["model"]["use_both_heads"]
        loss_lambda = self.config["model"]["loss_lambda"]
        
        for e in range(epochs):
            train_loss, train_acc, train_targets, train_predictions = utils.train_model(
                                                                                    self.model, 
                                                                                    self.data_loaders["train"], 
                                                                                    self.criterions, 
                                                                                    self.optimizer, 
                                                                                    device, 
                                                                                    use_both_heads,
                                                                                    loss_lambda,
                                                                      )
            
            
            test_loss, test_acc, test_targets, test_predictions = utils.test_model(
                                                                                self.model, 
                                                                                self.data_loaders["test"], 
                                                                                self.criterions, 
                                                                                device, 
                                                                                use_both_heads,
                                                                                loss_lambda,
                                                                  )
            
            print("EPOCH {}: Train acc: {:.2%} Train Loss: {:.4f} Test acc: {:.2%} Test Loss: {:.4f}".format( 
                                                                                                            e+1,
                                                                                                            train_acc,
                                                                                                            train_loss,
                                                                                                            test_acc,
                                                                                                            test_loss
                                                                                                        ))
            
                
            ### wandb logger
            metrics = {
                "train/train_loss": train_loss,
                "train/train_accuracy": train_acc,
                "test/test_loss": test_loss,
                "test/test_accuracy": test_acc,
              }

            wandb.log(metrics)
            
            wandb.log({"train/confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
                        y_true=train_targets, preds=train_predictions,
                        class_names=self.config["data"]["class_names"])})
            
            wandb.log({"test/confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
                        y_true=test_targets, preds=test_predictions,
                        class_names=self.config["data"]["class_names"])})

            
            ### local logger
            epoch_metrics = pd.DataFrame({
                "type": ['train', 'test'],
                "epoch": [e+1, e+1],
                "loss": [train_loss, test_loss],
                "accuracy": [train_acc, test_acc]
            })
            
            self.data_logger = pd.concat([self.data_logger, epoch_metrics], axis=0)
            
            ### lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            ### SAVE BEST MODEL
            if test_acc > self.best_accuracy:
                torch.save(self.model.state_dict(), self.path_artifact)
                self.best_accuracy = test_acc
                self.early_stopping_tolerance = self.config["hyperparameters"]["early_stopping_tolerance"]
                
                print(f"Saved best parameters at epoch {e+1}")
            else:
                self.early_stopping_tolerance -= 1
                if self.early_stopping_tolerance <= 0:
                    print(f"EARLY STOPPING AT ITERATION {e+1}")
                    break
                
            
        return self.model, self.data_logger

            
        