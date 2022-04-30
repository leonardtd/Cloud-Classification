import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms as T
import dgl
from dgl.nn import GATConv, GraphConv


from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class GCD:
    def __init__(self, image_paths, resize=None):

        self.image_paths  = image_paths
        self.targets = self._get_targets()
        
        self.resize = resize
        
        ### Extracted in notebook: GCD image stats
        self.meanR = 123.0767
        self.meanG = 156.9277
        self.meanB = 195.6296
        
        self.stdR = 22.0008
        self.stdG = 19.7520
        self.stdB = 17.5623
        
        self.norm_transform = T.Normalize(mean=[self.meanR, self.meanG, self.meanB], 
                    std=[self.stdR, self.stdG, self.stdB])
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item]).float()
        #Normalize by channel
        image = self.norm_transform(image)


        targets = self.targets[item]
        #substract 1 since list of targets start from 1
        targets = torch.tensor(targets, dtype=torch.long) - 1

        if self.resize is not None:
            image = T.Resize(self.resize)(image)

        return {
            "images": image,
            "targets": targets,
            'paths': self.image_paths[item].split('/')[-1]
        }
    
    
    
    def _get_targets(self):
        return list(map(int,list(map(int,[os.path.basename(x).split('_')[0] 
                                          for x in self.image_paths]))))
    
    

def train(model, data_loader, criterion, optimizer, device):
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
    
    
def test(model, data_loader, criterion, device):
    model.eval()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                if k!='paths':
                    data[k] = v.to(device)

            logits = model(data["images"])
            loss = criterion(logits, data["targets"])
            fin_loss += loss.item()

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())

    return (
        np.concatenate(fin_preds,axis=0),
        np.concatenate(fin_targs,axis=0),
        fin_loss / len(data_loader),
    )


### GRAD VISUAL


################# MODELS ########################

class ResNet50(nn.Module):
    def __init__(self, out_channels, feature_extraction):
        super().__init__()
        
        self.cnn = models.resnet50(pretrained=True)
        
        ### Freeze params and initialize
        if feature_extraction:
            for param in self.parameters():
                param.requires_grad = False
        
        self.cnn.fc = nn.Linear(2048, out_channels)
        
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.cnn.fc.weight, gain=gain)
        self.cnn.fc.bias.data.fill_(0)
        
    def forward(self, x):
        return self.cnn(x)
    
    
class CNNExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )
        
    def forward(self, x):
        return self.cnn(x).view(-1,2048)

    
class GraphLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = GraphConv(in_channels, out_channels)
        
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=gain)
        self.conv.bias.data.fill_(0)
        
    def forward(self, g, x):
        return self.conv(g, x)
    
class GraphClassifier(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        self.THRESHOLD = 0.7
        
        self.cnn = CNNExtractor()
        self.graphconv = GraphLayers(2048, 512)
        
        self.mlp = nn.Sequential(
            nn.Linear(2048+512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels)
        )
        
        self.mlp.apply(self.init_weights)
        
        
    def init_weights(self, m):
        
        gain = nn.init.calculate_gain('relu') 
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)
            
    def get_deep_features(self, x):
        return self.cnn(x)
    
    def message_passing(self, deep_features):
        ### CREATE GRAPH
        with torch.no_grad():
            norm = deep_features.norm(dim=1).view(-1,1)
            batch_nodes = deep_features/norm

            sim_matrix = batch_nodes @ batch_nodes.T
            adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
            row, col = torch.where(adj_matrix==1)

            g = dgl.graph((row, col))
        
        agg_features = self.graphconv(g, deep_features)
        final_features = torch.cat([deep_features, agg_features], dim=1)
                
        return self.mlp(final_features)
        
        
    def get_adjacency_matrix(self, x):
        with torch.no_grad():
            deep_features = self.cnn(x)
            norm = deep_features.norm(dim=1).view(-1,1)
            batch_nodes = deep_features/norm

            sim_matrix = batch_nodes @ batch_nodes.T
            adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
        
        return adj_matrix.cpu(), sim_matrix.cpu()
    
        
    def forward(self, x):
        deep_features = self.cnn(x)
        
        ### CREATE GRAPH
        with torch.no_grad():
            norm = deep_features.norm(dim=1).view(-1,1)
            batch_nodes = deep_features/norm

            sim_matrix = batch_nodes @ batch_nodes.T
            adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
            row, col = torch.where(adj_matrix==1)

            g = dgl.graph((row, col))
        
        agg_features = self.graphconv(g, deep_features)
        final_features = torch.cat([deep_features, agg_features], dim=1)
                
        return self.mlp(final_features)
    
    
    

class GraphClassifier2(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        
        self.THRESHOLD = 0.7
        
        self.cnn = CNNExtractor()
        self.graphconv = GraphLayers(2048, 512)
        
        self.mlp = nn.Sequential(
            nn.Linear(2048+512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels)
        )
        
        self.mlp.apply(self.init_weights)
        
        
    def init_weights(self, m):
        
        gain = nn.init.calculate_gain('relu') 
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)
        
    def get_adjacency_matrix(self, x):
        with torch.no_grad():
            deep_features = self.cnn(x)
            norm = deep_features.norm(dim=1).view(-1,1)
            batch_nodes = deep_features/norm

            sim_matrix = batch_nodes @ batch_nodes.T
            adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
        
        return adj_matrix.cpu(), sim_matrix.cpu()
    
        
    def forward(self, x):
        deep_features = self.cnn(x)
        
        ### CREATE GRAPH
        with torch.no_grad():
            norm = deep_features.norm(dim=1).view(-1,1)
            batch_nodes = deep_features/norm

            sim_matrix = batch_nodes @ batch_nodes.T
            adj_matrix = torch.where(sim_matrix > self.THRESHOLD, 1, 0)
            row, col = torch.where(adj_matrix==1)

            g = dgl.graph((row, col))
        
        agg_features = self.graphconv(g, deep_features)
        final_features = torch.cat([deep_features, 0.7*agg_features], dim=1)
                
        return self.mlp(final_features)
        
        
