#!/usr/bin/env python
# coding: utf-8


import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges #depricated




dataset = Planetoid("./data", "CiteSeer", transform = T.NormalizeFeatures())



dataset.data # X = [3327, 3703] for each node we have 3703 labels



data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data




data = train_test_split_edges(data)



data = T.RandomLinkSplit(data)




data



type(data)


# neg_edge -> edges that are not in graph , pos_edge -> edges that are present in graph

# ## Define The Encoder



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__() # in this case we have only one graph, it is useful to cache
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True) #cached only for transductive learning - caches the normalization of the adjacences matrices
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True) #cached only for transductive learning
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# ## Define The AutoEncoder



from torch_geometric.nn import GAE




# paramets
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model 
model = GAE(GCNEncoder(num_features, out_channels))

#move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.num_val.x.to(device)
train_pos_edge_index = data.num_val.train_pos_edge_index.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




model




#dir(model)




def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)
    




for epoch in range(1, epochs+1):
    loss = train()
    
    auc, ap = test(data.num_val.test_pos_edge_index, data.num_val.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))




Z = model.encode(x, train_pos_edge_index)
Z


# ## Use Tensorboard



from torch.utils.tensorboard import SummaryWriter



# paramets
out_channels = 20
num_features = dataset.num_features
epochs = 1000

# model 
model = GAE(GCNEncoder(num_features, out_channels))

#move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.num_val.x.to(device)
train_pos_edge_index = data.num_val.train_pos_edge_index.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




writer = SummaryWriter('runs/GAE_experiment_'+'20d_1000_epochs')




for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.num_val.test_pos_edge_index, data.num_val.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    
    writer.add_scalar('auc train',auc,epoch) # new line
    writer.add_scalar('ap train',ap,epoch)   # new line

'''
# ## Graph Variational AutoEncoder (GVAE)


from torch_geometric.nn import VGAE




dataset = Planetoid("./data", "CiteSeer", transform = T.NormalizeFeatures())
data2 = dataset[0]
data2.train_mask = data2.val_mask = data2.test_mask = data2.y = None
data2 = T.RandomLinkSplit(data2)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cache only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)




out_channels = 2
num_features = dataset.num_features
epochs = 100

model = VGAE(VariationalGCNEncoder(num_features, out_channels)) #instantiate the VGAE by passing encoder

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

model = model.to(device)
x = data2.num_val.x.to(device)

# train_pos_edge_index = data.train_pos_edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, data2.num_val.train_pos_edge_index)
    loss = model.recon_loss(z, data2.num_val.train_pos_edge_index)
    
    loss = loss + (1 / data2.num_val.num_nodes) * model.kl_loss() #new_line
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, data2.num_val.train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    




writer = SummaryWriter('runs/VGAE_experiment_'+'2d_100_epochs')

for epoch in range(1, epochs+1):
    loss = train()
    auc, ap = test(data2.num_val.test_pos_edge_index, data2.num_val.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    
    writer.add_scalar('auc_train', auc, epoch)
    writer.add_scalar('ap_train', ap, epoch)

'''




