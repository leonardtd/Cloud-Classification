import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv, GraphConv

from .conv_modules import CNNExtractor


def normalize_features(dfs):
    norm = dfs.norm(dim=1).view(-1,1)
    return dfs/norm

###########################
#   ADJACENCY BUILDERS
###########################

def build_graph_cosine_similarity(deep_features, threshold):
    """
    returns a dgl graph built with an edge index calculated from the deep features 
    using cosine similarity
    """
    
    with torch.no_grad():
        batch_nodes = normalize_features(deep_features)

        sim_matrix = batch_nodes @ batch_nodes.T
        adj_matrix = torch.where(sim_matrix > threshold, 1, 0)

        row, col = torch.where(adj_matrix==1)
 
    return dgl.graph((row, col))


def build_graph_pearson_correlation(dfs, threshold):
    """
    returns a dgl graph built with an edge index calculated from the deep features 
    using pearson correlation
    """
    with torch.no_grad():
        corr_matrix = torch.corrcoef(dfs)
        adj_matrix = torch.where(corr_matrix > threshold, 1, 0)
        row, col = torch.where(adj_matrix==1)
    
    return dgl.graph((row, col))


###########################
#   MODULES
###########################

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = GraphConv(in_channels, out_channels)
                
    def forward(self, g, x):
        return self.conv(g, x)
    
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, residual, agg='mean'):
        super().__init__()
        
        self.conv = GATConv(in_channels, out_channels, num_heads, residual)
        self.agg = agg
        
    def forward(self, g, x):
        x = self.conv(g, x)
        if self.agg == 'mean':
            return x.mean(dim=1)
        elif self.agg == 'sum':
            return x.sum(dim=1)
        else: return x
    

class GraphClassifier(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_hidden, 
                 num_classes,
                 conv_type,
                 conv_parameters: dict,
                 adjacency_builder,
                 builder_parameter,
                 use_both_heads = True,
                ):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.conv_parameters = conv_parameters
        self.adjacency_builder = adjacency_builder
        self.builder_parameter = builder_parameter
        self.use_both_heads = use_both_heads
        
        ### modules
        self.cnn = CNNExtractor() # 2048 output dims
        
        self.graph_layers = self.build_graph_layers(hidden_dim, num_hidden, conv_type, conv_parameters)

        self.head = nn.Sequential(
                    nn.Linear(2048 + hidden_dim, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, num_classes),
                )
        
        self.second_head = nn.Linear(2048, num_classes) if use_both_heads else nn.Identity()
        
        self.reset_parameters()
        
    def build_graph_layers(self, hidden_dim, num_hidden, conv_type, conv_parameters):
        
        module_list = nn.ModuleList()
        
        if conv_type == 'gcn':
            module_list.append(GraphConvLayer(2048, hidden_dim))
            conv = GraphConvLayer(hidden_dim, hidden_dim)
        elif conv_type == 'gat':
            module_list.append(GraphAttentionLayer(2048, hidden_dim, conv_parameters["num_heads"], conv_parameters["residual"], conv_parameters['agg']))
            conv = GraphAttentionLayer(hidden_dim, hidden_dim, conv_parameters["num_heads"], conv_parameters["residual"], conv_parameters['agg'])
        else:
            raise NotImplementedError("Invalid layer")
        
        for i in range(num_hidden-1):
            module_list.append(conv)
            
        return module_list
        
        
    def reset_parameters(self):
        for m in self.modules():
            if  isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                

    def forward(self, x):
        
        deep_features = self.cnn(x)
        
        ### ADJACENCY MATRIX CONSTRUCTION
        # TODO: implement distance and mlp
        
        if self.adjacency_builder == 'cos_sim':
            g = build_graph_cosine_similarity(deep_features.detach(), self.builder_parameter)
        elif self.adjacency_builder == 'pearson_corr':
            g = build_graph_pearson_correlation(deep_features.detach(), self.builder_parameter)
        else:
            raise NotImplementedError("Invalid builder")
    
        ### MESSAGE PASSING
        x = deep_features
        
        for i, gnn_layer in enumerate(self.graph_layers):
            x = gnn_layer(g, x)
            if i != len(self.graph_layers)-1:
                x = F.leaky_relu(x)
        
        agg_features = torch.cat([deep_features, x], dim=1)
        logits_agg = self.head(agg_features)
        
        logits_cnn = self.second_head(deep_features)

        return logits_agg, logits_cnn
    
    def get_deep_features(self, x):
        with torch.no_grad():
            features = self.cnn(x)
            
        return features
    