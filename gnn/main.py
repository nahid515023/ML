import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model parameters
input_dim = dataset.num_node_features  # Number of input features
hidden_dim = 16                        # Hidden layer size
output_dim = dataset.num_classes       # Number of classes

# Initialize the model, optimizer, and loss function
model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    output = model(dataset[0])
    loss = criterion(output[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    output = model(dataset[0])
    pred = output.argmax(dim=1)
    correct = pred[dataset[0].test_mask] == dataset[0].y[dataset[0].test_mask]
    acc = int(correct.sum()) / int(dataset[0].test_mask.sum())
    return acc

# Train the model
epochs = 200
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Final evaluation
final_acc = test()
print(f'Final Test Accuracy: {final_acc:.4f}')
