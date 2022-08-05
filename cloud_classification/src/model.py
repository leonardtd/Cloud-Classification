from . import utils
from .modules import *


def build_model()

class ModelTrainer:
    def __init__(self, config, use_wandb):
        self.use_wandb = use_wandb
        self.config = config
        
    def train(self):
        
        model = build_model(config["graph_builder"], config["num_classes"])