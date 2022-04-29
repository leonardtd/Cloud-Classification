from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import config


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
            
    plt.figure(figsize=(100,100))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    #plt.savefig('images/grads.png')

def forward_backward_pass(model, data_loader, criterion, optimizer, device):
    model.train()
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        
        logits = model(data["images"])
        
        _loss = criterion(logits, data["targets"])
        _loss.backward()
        
        optimizer.step()
        
    return _loss.item()

def train_fn(model, data_loader, criterion, optimizer, device):
    model.train()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
      
        logits = model(data["images"])
        
        _loss = criterion(logits, data["targets"])
        _loss.backward()
        
        #plot_grad_flow(model.named_parameters())
        
        optimizer.step()
        
        fin_loss += _loss.item()

        batch_preds = F.softmax(logits, dim=1)
        batch_preds = torch.argmax(batch_preds, dim=1)

        fin_preds.append(batch_preds.cpu().numpy())
        fin_targs.append(data["targets"].cpu().numpy())

    return (np.concatenate(fin_preds,axis=0), 
            np.concatenate(fin_targs,axis=0), 
            fin_loss / len(data_loader))


def eval_fn(model, data_loader, criterion, device):
    model.eval()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            logits = model(data["images"])
            _loss = criterion(logits, data["targets"])
            fin_loss += _loss.item()

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())

    return (
        np.concatenate(fin_preds,axis=0),
        np.concatenate(fin_targs,axis=0),
        fin_loss / len(data_loader),
    )
