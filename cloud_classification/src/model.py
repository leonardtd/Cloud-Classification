import wandb

from . import utils
from .modules import GraphClassifier


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
        self.scheduler = utils.build_scheduler(self.config) if config["hyperparameters"]["use_scheduler"] else None
        
        
        if self.use_wandb:
            wandb.watch(self.model, log="all")
            
    def train(self, model, data_loader, criterion, optimizer, device):
        ### TODO
        model.train()

        fin_loss = 0
        fin_preds = []
        fin_targs = []

        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                if k!='paths':
                    data[k] = v.to(device)

            optimizer.zero_grad()

            logits = model(data["images"])

            loss = criterion(logits, data["targets"])
            loss.backward()

            optimizer.step()

            fin_loss += loss.item()

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())

        return (np.concatenate(fin_preds,axis=0), 
                np.concatenate(fin_targs,axis=0), 
                fin_loss / len(data_loader))
        
        