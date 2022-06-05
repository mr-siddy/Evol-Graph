#!/usr/bin/env python
# coding: utf-8


import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.loader as loader
from torch_geometric.nn.inits import uniform
from torch.nn import Parameter as Param
from torch import Tensor
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn.conv import MessagePassing


dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit('train_rest', num_val=500, num_test=  500),
    T.TargetIndegree(),
])

dataset = Planetoid('./data', dataset, pre_transform = T.NormalizeFeatures(), transform = transform)
data = dataset[0]
data = data.to(device)


# ## Transition and Output Function Instantiation



class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.Tanh())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


# ## GNN Module - State Propagations and Node States



class GNNM(MessagePassing):
    def __init__(self, n_nodes, out_channels, features_dim, hid_dims, num_layers = 50, eps=1e-3, aggr = 'add',
                 bias = True, **kwargs):
        super(GNNM, self).__init__(aggr=aggr, **kwargs)

        self.node_states = Param(torch.zeros((n_nodes, features_dim)), requires_grad=False)
        self.out_channels = out_channels
        self.eps = eps
        self.num_layers = num_layers
        
        self.transition = MLP(features_dim, hid_dims, features_dim)
        self.readout = MLP(features_dim, hid_dims, out_channels)
        
        self.reset_parameters()
        print(self.transition)
        print(self.readout)

    def reset_parameters(self):
        self.transition.reset_parameters()
        self.readout.reset_parameters()
        
    def forward(self): 
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        node_states = self.node_states
        for i in range(self.num_layers):
            m = self.propagate(edge_index, x=node_states, edge_weight=edge_weight,
                               size=None)
            new_states = self.transition(m)
            with torch.no_grad():
                distance = torch.norm(new_states - node_states, dim=1)
                convergence = distance < self.eps
            node_states = new_states
            if convergence.all():
                break
            
        out = self.readout(node_states)
        
        return F.log_softmax(out, dim=-1)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)




model = GNNM(data.num_nodes, dataset.num_classes, 32, [64,64,64,64,64], eps=0.01).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = loader.DataLoader(test_dataset)
train_loader = loader.DataLoader(train_dataset)

def train():
    model.train()
    optimizer.zero_grad()
    loss_fn(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 51):
    train()
    accs = test()
    train_acc = accs[0]
    val_acc = accs[1]
    test_acc = accs[2]
    print('Epoch: {:03d}, Train Acc: {:.5f}, '
          'Val Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, train_acc,
                                                       val_acc, test_acc))






