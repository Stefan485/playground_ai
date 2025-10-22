# fake_news_gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import networkx as nx


class FakeNewsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Placeholder – assume raw graphs exist under raw/
        return ["./dataset/gossipcop_fake.csv", "./dataset/gossipcop_true.csv", "./dataset/politifact_fake.csv", "./dataset/politifact_fake.csv",]

    @property
    def processed_file_names(self):
        return ["data.pt"]


    def process(self):
        data_list = []

        # ======== MOCK DATA EXAMPLE ========
        # When using FakeNewsNet, you’d loop over each news item,
        # build a NetworkX graph from retweets and extract features.
        for i in range(100):  # 100 fake graphs for now
            num_nodes = np.random.randint(10, 50)
            G = nx.erdos_renyi_graph(num_nodes, 0.2)

            # Node features (e.g. random or text embeddings)
            x = torch.randn((num_nodes, 16))  # 16-dim features

            # Edge index from NetworkX
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            if edge_index.numel() == 0:  # empty graph fallback
                continue

            # Graph label (0=fake, 1=real)
            y = torch.tensor([np.random.randint(0, 2)], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class FakeNewsGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # First graph convolution + activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second convolution + dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Pool node embeddings to graph embedding
        x = global_mean_pool(x, batch)

        # Final linear classifier
        return self.lin(x)


# =====================
# 3. Training Loop
# =====================

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def main():
    dataset = FakeNewsDataset(root="./data/fakenews")

    # Split dataset
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_dataset = dataset[:80]
    test_dataset = dataset[80:]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeNewsGCN(in_channels=16, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1, 51):
        loss = train(model, train_loader, optimizer, device)
        acc = test(model, test_loader, device)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")


if __name__ == "__main__":
    main()
