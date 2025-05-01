# gnn_test.py
#heloooo?
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 1. Load the Cora dataset (automatically downloads to ./data/Cora)
dataset = Planetoid(root='data/Cora', name='Cora')

# 2. Define a 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Instantiate model, optimizer, data
device = torch.device('cpu')  # or 'mps' for Apple GPU if configured
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. Training loop
model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d}  Loss: {loss:.4f}')

# 5. Evaluation
model.eval()
out = model(data)
pred = out.argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
print(f'\nTest Accuracy: {acc:.4f}')