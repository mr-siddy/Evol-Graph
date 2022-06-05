#!/usr/bin/env python
# coding: utf-8

# # Node level prediction on Cora using GCN
# - toc: True
# - categorites: [Graph Machine Learning] 

# ## Data Loading and Preprocessing



import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import torch_geometric.loader as loader
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




name = 'Cora'
transform = transforms.Compose([
    transforms.RandomNodeSplit('train_rest', num_val=500, num_test=500),
    transforms.TargetIndegree()
])
cora = datasets.Planetoid('./data', name, pre_transform=transforms.NormalizeFeatures(), transform=transform)
# pre_transform applied only when the dataset is downloading
# once dataeset is downloaded the transforms will be applies 
# if dataset is already downloaded the running cell again will retrive dataset from the local it self




print("Cora info:")
print("# of graphs:", len(cora))
print("# Classes {graphs}", cora.num_classes)
print("# Edge features", cora.num_edge_features)
print("# Node features", cora.num_node_features)




print(cora.data)



print("edge_index:", cora.data.edge_index.shape)
print(cora.data.edge_index)
print("\n")
print("train_mask", cora.data.train_mask.shape)
print(cora.data.train_mask)
print("\n")
print("x:", cora.data.x.shape)
print(cora.data.x)
print("\n")
print("y:", cora.data.y.shape)
print(cora.data.y)




first = cora[0]
print("Number of nodes: ", first.num_nodes)
print("Number of edges: ", first.num_edges)
print("Is directed: ", first.is_directed())




print("Shape of sample nodes: ", first.x[:5].shape)



print("# of nodes to train on: ", first.train_mask.sum().item())

print("# of nodes to test on: ", first.test_mask.sum().item())

print("# of nodes to validate on: ", first.val_mask.sum().item())




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## GCN model preparation


device




data = cora[0].to(device)




print("X shape: ", data.x.shape)
print("Edge shape: ", data.edge_index.shape)
print("Y shape: ", data.y.shape)




class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, cora.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)




model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)




def compute_accuracy(pred_y, y):
    return (pred_y == y).sum()




model.train()
losses = []
accuracies = []
for epoch in range(400):
    optimizer.zero_grad()
    out = model(data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    correct = compute_accuracy(out.argmax(dim=1)[data.train_mask], data.y[data.train_mask])
    acc = int(correct) / int(data.train_mask.sum())
    losses.append(loss.item())
    accuracies.append(acc)

    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))


# ## Training Results



import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(accuracies)
plt.legend(['Loss', 'Accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy v/s No. of epochs")
plt.show()



model.eval()
pred = model(data).argmax(dim=1)
correct = compute_accuracy(pred[data.test_mask], data.y[data.test_mask])
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')






