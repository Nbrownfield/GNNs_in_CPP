import os
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import pandas as pd

from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj

dataset = KarateClub()

data = dataset[0]

# Graph Convolutional Network
class GCN(torch.nn.Module):

    #stacks 3 convolution layers, aggregating 3-hop neighborehood info around each node (each node gets info from all nodes up to 3 hops away)
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4, bias=False)
        self.conv2 = GCNConv(4, dataset.num_classes, bias=False)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        out = self.conv2(h, edge_index)

        return out, h

model = GCN()

criterion = torch.nn.CrossEntropyLoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Initialize the Adam optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(400):
    loss, h = train(data)
    print(f'Epoch: {epoch}, Loss: {loss}')

#write weights out to .csv files
paramIndex = 0
for name, param in model.named_parameters():
    print(name, torch.transpose(param, 0, 1))

    paramIndex = paramIndex + 1

    filename = "Weights_conv" + str(paramIndex) + ".csv"
    t_np = torch.transpose(param, 0, 1).detach().numpy()
    df = pd.DataFrame(t_np)
    df.to_csv(filename,index=False, header=False)

#write adj matrix
adjMat = to_dense_adj(data.edge_index).reshape([dataset.num_features,dataset.num_features]) #convert from 3d ([1,34,34]) tensore to 2d ([34,34])
adjMatNp = adjMat.numpy()
adjDf = pd.DataFrame(adjMatNp)
adjDf.to_csv("adjMat.csv",index=False, header=False)

#write x node feature matrix
xNp = data.x.numpy()

xDf = pd.DataFrame(xNp)
xDf.to_csv("xMat.csv",index=False, header=False)

out, h = model(data.x, data.edge_index)
print(out[data.train_mask])
print(data.train_mask)
print(data.y[data.train_mask])
