#!/usr/bin/env python
# coding: utf-8

# 
# # GAE for Link Prediction


from sklearn.metrics import roc_auc_score

import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




name = 'Cora'
dataset = Planetoid('./data', name, transform=T.NormalizeFeatures())




data = dataset[0]
print(dataset.data)



#use random_link_split to create neg and pos edges
data.train_mask = data.val_mask = data.test_mask = data.y = None

data = train_test_split_edges(data)
#data = T.RandomLinkSplit(data, add_negative_train_samples=True)
print(data)


# ## Simple Autoencoder Model



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)
        
    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index) #conv1
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index) #conv2
    
    def decode(self, z, pos_edge_index, neg_edge_index): #only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1) #concat pos and neg edge
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1) #dot product
        return logits
    
    def decode_all(self, z):
        prob_adj = z @ z.t() #get adj NxN
        return (prob_adj >0).nonzero(as_tuple = False).t() #get predicted edge list




model, data = Net().to(device), data.to(device)




optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




'''
return array sequence of ones equal to the length for pos_edges_index /
and sequence of 0 equal to the length of neg_edge_index
   ''' 
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

def train():
    model.train()
    
    neg_edge_index = negative_sampling( #PyG Func
        edge_index = data.train_pos_edge_index,
        num_nodes = data.num_nodes,
        num_neg_samples = data.train_pos_edge_index.size(1) # num of neg_samples = num of pos_samples
    )
    
    optimizer.zero_grad()
    z = model.encode() # encoding
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index) #decode
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test():
    model.eval()
    performance = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        
        performance.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return performance




best_val_perf = test_perf = 0
for epoch in range(1, 101):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Test: {:.4f}'
    if epoch % 10 == 0:
        print(log.format(epoch, train_loss, best_val_perf, test_perf))




z = model.encode()



final_edge_embedding = model.decode_all(z)
final_edge_embedding






