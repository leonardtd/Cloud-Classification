from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from . import config



def forward_backward_pass(model, data_loader, criterion, optimizer, device):
    model.train()
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        
        logits_1, logits_2 = model(data["images"])
        
        _loss_1 = criterion(logits_1, data["targets"])
        _loss_2 = criterion(logits_2, data["targets"])
        _loss = _loss_1 + config.GNN_LOSS_LAMBDA*_loss_2
        
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
        
        logits_1, logits_2 = model(data["images"])
        
        _loss_1 = criterion(logits_1, data["targets"])
        _loss_2 = criterion(logits_2, data["targets"])
        _loss = _loss_1 + config.GNN_LOSS_LAMBDA*_loss_2
        
        _loss.backward()
        
        optimizer.step()
        
        fin_loss += _loss.item()

        batch_preds = F.softmax(logits_1, dim=1)
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

            logits_1, logits_2 = model(data["images"])
            _loss_1 = criterion(logits_1, data["targets"])
            _loss_2 = criterion(logits_2, data["targets"])
            _loss = _loss_1 + config.GNN_LOSS_LAMBDA*_loss_2
            
            fin_loss += _loss.item()

            batch_preds = F.softmax(logits_1, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())

    return (
        np.concatenate(fin_preds,axis=0),
        np.concatenate(fin_targs,axis=0),
        fin_loss / len(data_loader),
    )


def train_fn_new(model, data_loader, criterion, optimizer, device):
    model.train()
    fin_loss = 0
    
    deep_features = []
    targets = []
    
    gnn_logits = []
    cnn_logits = []
    

    for data in tqdm(data_loader, total=len(data_loader)):
        
        with torch.autograd.graph.save_on_cpu():
            x = model.get_deep_features(data["images"].to(device))
        
        deep_features.append(x.cpu())
        #targets.append(data["targets"].cpu())
    
    
#     deep_features = torch.cat(deep_features, dim=0).to(device)
#     targets = torch.cat(targets, dim=0).to(device)
    
#     logits_gnn, logits_cnn = model.message_passing(deep_features)
    
#     optimizer.zero_grad()
    
#     loss_1 = criterion(logits_gnn, targets)
#     loss_2 = criterion(logits_cnn, targets)
#     loss = loss_1 + config.GNN_LOSS_LAMBDA*loss_2
    
#     loss.backward()
    
#     optimizer.step()

#     fin_loss = loss.item()

#     preds = F.softmax(logits_gnn, dim=1)
#     preds = torch.argmax(preds, dim=1)

#     fin_preds = preds.cpu().numpy()
#     fin_targs = targets.cpu().numpy()

#     return fin_preds, fin_targs, fin_loss



def eval_fn_large(model, data_loader, criterion, device):
    model.train()
    fin_loss = 0
    
    deep_features = []
    targets = []
    
    gnn_logits = []
    cnn_logits = []
    

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):

            x = model.get_deep_features(data["images"].to(device))

            deep_features.append(x.cpu())
            targets.append(data["targets"].cpu())


        deep_features = torch.cat(deep_features, dim=0).to(device)
        targets = torch.cat(targets, dim=0).to(device)

        logits_gnn, logits_cnn = model.message_passing(deep_features)

        loss_1 = criterion(logits_gnn, targets)
        loss_2 = criterion(logits_cnn, targets)
        loss = loss_1 + config.GNN_LOSS_LAMBDA*loss_2

        fin_loss = loss.item()

        preds = F.softmax(logits_gnn, dim=1)
        preds = torch.argmax(preds, dim=1)

        fin_preds = preds.cpu().numpy()
        fin_targs = targets.cpu().numpy()

        return fin_preds, fin_targs, fin_loss