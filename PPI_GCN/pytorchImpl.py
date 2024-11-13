import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas as pd
import time

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print('Dataset properties')
print('==============================================================')
print(f'Dataset: {train_dataset}') #This prints the name of the dataset
print(f'Number of graphs in the dataset: {len(train_dataset)}')
print(f'Number of features: {train_dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of classes: {train_dataset.num_classes}') #Number of classes that a node can be classified into

print(f'Number of test graphs: {len(test_dataset)}') #Number of classes that a node can be classified into

for data in test_loader:
    print(data.edge_index)
    print(f'Number of nodes in graph: {data.num_nodes}')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(train_dataset.num_features, 64, bias=False)
        self.conv2 = GCNConv(64, train_dataset.num_classes, bias=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cpu')
model = GCN().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)

        start_time = time.perf_counter_ns()
        out = model(data.x.to(device), data.edge_index.to(device))
        end_time = time.perf_counter_ns()

        elapsed_time_ns = end_time - start_time
        elapsed_time_us = elapsed_time_ns / 1000
        print(f'Time taken for one pass of model: {elapsed_time_us} microseconds')
        
        preds.append((out > 0.5).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

#train model
times = []
for epoch in range(1, 141):
    start = time.time()
    loss = train()
    #val_f1 = test(val_loader)
    #test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

#write weights out to .csv files
paramIndex = 0
for name, param in model.named_parameters():
    paramIndex = paramIndex + 1
    filename = "data/Weights_conv" + str(paramIndex) + ".csv"

    t_np = torch.transpose(param, 0, 1).detach().numpy()
    df = pd.DataFrame(t_np)
    df.to_csv(filename,index=False, header=False)

#write adj matrices
adjMatIndex = 0
for data in test_loader:
    adjMatIndex = adjMatIndex + 1
    filename = "data/adjMat" + str(adjMatIndex) + ".csv"

    adjMat = to_dense_adj(data.edge_index)
    adjMat = torch.reshape(adjMat, [adjMat.size(1),adjMat.size(2)]) #convert from 3d tensor to 2d
    adjMatNp = adjMat.numpy()
    adjDf = pd.DataFrame(adjMatNp)
    adjDf.to_csv(filename,index=False, header=False)

#Write X node feature matrices
xIndex = 0
for data in test_loader:
    xIndex = xIndex + 1
    filename = "data/xMat" + str(xIndex) + ".csv"

    xNp = data.x.numpy()
    xDf = pd.DataFrame(xNp)
    xDf.to_csv(filename,index=False, header=False)

#Write Y expected output matrices

yIndex = 0
for data in test_loader:
    yIndex = yIndex + 1
    filename = "data/yMat" + str(xIndex) + ".csv"

    yNp = data.y.numpy()
    yDf = pd.DataFrame(yNp)
    yDf.to_csv(filename,index=False, header=False)

test_f1 = test(test_loader)
print(test_f1)