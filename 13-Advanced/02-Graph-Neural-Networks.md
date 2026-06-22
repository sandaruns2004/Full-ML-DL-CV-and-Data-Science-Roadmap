# 🕸️ Graph Neural Networks (GNNs)

> **Prerequisites**: Neural Networks, Linear Algebra | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Limitation of Grids and Sequences](#1-the-limitation-of-grids-and-sequences)
2. [Graph Theory Fundamentals](#2-graph-theory-fundamentals)
3. [Message Passing Framework](#3-message-passing-framework)
4. [Graph Convolutional Networks (GCN)](#4-graph-convolutional-networks-gcn)
5. [Graph Attention Networks (GAT)](#5-graph-attention-networks-gat)
6. [Library Implementation (PyTorch Geometric)](#6-library-implementation-pytorch-geometric)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Limitation of Grids and Sequences

Deep Learning has conquered images using CNNs and text using Transformers/RNNs. 
- **Images** are perfectly structured 2D grids of pixels.
- **Text** is a perfectly structured 1D sequence of words.

However, the real world is not a grid. Much of the world's most valuable data exists in non-Euclidean spaces as **Graphs**:
- **Social Networks**: People are nodes, friendships are edges.
- **Chemistry**: Atoms are nodes, chemical bonds are edges.
- **Traffic/Maps**: Cities are nodes, roads are edges.
- **Recommendation Systems**: Users and Products are nodes, purchases are edges.

Standard Neural Networks cannot process this data because the number of connections for each node varies wildly, and there is no strict spatial order (up/down/left/right). We need **Graph Neural Networks (GNNs)**.

---

## 2. Graph Theory Fundamentals

A Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ consists of:
- **Vertices (Nodes)** $\mathcal{V}$: The entities.
- **Edges** $\mathcal{E}$: The connections between entities.

To feed a graph into a neural network, we represent it using two matrices:
1. **Node Feature Matrix ($\mathbf{X}$)**: A matrix of size $N \times F$, where $N$ is the number of nodes, and $F$ is the number of features per node. (e.g., if nodes are people, features could be [age, salary, height]).
2. **Adjacency Matrix ($\mathbf{A}$)**: An $N \times N$ matrix. $A_{i,j} = 1$ if there is an edge between node $i$ and node $j$, and $0$ otherwise.

---

## 3. Message Passing Framework

The core concept behind all modern GNNs is **Message Passing**.

If we want to predict if Node $A$ is a fraudulent bank account, we shouldn't just look at Node $A$'s features. We should look at the features of all the accounts Node $A$ has sent money to (its neighbors).

**The Message Passing Step:**
At layer $k$, the hidden state (feature vector) of node $v$, denoted as $\mathbf{h}_v^{(k)}$, is updated using three operations:
1. **Message**: Extract features from all neighboring nodes $\mathcal{N}(v)$.
2. **Aggregate**: Combine the neighbor features into a single vector (using a permutation-invariant function like SUM, MEAN, or MAX).
3. **Update**: Combine the aggregated neighbor features with node $v$'s *own* previous features, pass them through a Neural Network layer, and apply an activation function (like ReLU).

$$ \mathbf{h}_v^{(k)} = \text{Update}^{(k)}\left( \mathbf{h}_v^{(k-1)}, \text{Aggregate}^{(k)}\left( \{ \mathbf{h}_u^{(k-1)} \}_{u \in \mathcal{N}(v)} \right) \right) $$

If you stack $K$ GNN layers, each node learns information from its $K$-hop neighbors!

---

## 4. Graph Convolutional Networks (GCN)

In 2016, Kipf & Welling introduced the **GCN**, bringing the power of Convolutions to Graphs.

In a CNN, a convolution is a weighted sum of a central pixel and its neighboring grid pixels.
In a GCN, a convolution is a weighted sum of a central node and its connected neighbor nodes.

### The GCN Mathematics
Instead of looping over every node (which is slow), GCNs use highly optimized matrix multiplication:

$$ \mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right) $$

Let's break this massive equation down:
1. $\mathbf{H}^{(l)}$: The node features at the current layer.
2. $\mathbf{W}^{(l)}$: The learnable weight matrix (just like a standard linear layer).
3. $\tilde{\mathbf{A}}$: The Adjacency Matrix **with self-loops added** (we add the Identity matrix so that nodes don't forget their own features when aggregating).
4. $\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}}$: This is the **Symmetric Normalized Adjacency Matrix**. We must normalize $\tilde{\mathbf{A}}$ by the node degrees $\mathbf{D}$. Without normalization, popular nodes (with 1000 edges) would have exploding feature vectors simply because they sum up 1000 neighbors, while isolated nodes would vanish.

---

## 5. Graph Attention Networks (GAT)

A GCN treats all neighbors equally. If Node $A$ has 10 friends, it takes the exact mathematical average of all 10 friends.

But in reality, some connections are more important than others!
In 2017, the **Graph Attention Network (GAT)** was invented. It applies the Transformer's Attention mechanism to Graphs.

Instead of a static Adjacency Matrix, GAT learns **Attention Weights** $\alpha_{i,j}$ between connected nodes $i$ and $j$.
When Node $i$ aggregates information from its neighbors, it performs a *weighted* sum, where the weights are dynamically calculated by an Attention neural network layer.

$$ \mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)} \right) $$

*(Like Transformers, GATs use Multi-Head Attention to stabilize learning).*

---

## 6. Library Implementation (PyTorch Geometric)

Implementing GNNs from scratch with sparse matrix operations is painful. The industry standard library is **PyTorch Geometric (PyG)**.

Let's build a GCN for Node Classification on the famous Cora dataset (predicting the category of a scientific paper based on its citation network).

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 1. Load the Cora Citation Network Dataset
# Nodes = Documents, Edges = Citations, Node Features = Bag of Words
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 2. Define the GCN Architecture
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        # GCNConv expects (in_channels, out_channels)
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        # x is the Node Feature matrix (N, F)
        # edge_index is a sparse (2, E) tensor listing all connected node pairs
        x, edge_index = data.x, data.edge_index

        # Layer 1: Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2: Final message passing and classification
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Setup Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. Training Loop (Node Classification)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # We only calculate loss on the small percentage of nodes that are labeled!
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Test Accuracy: {acc:.4f}') # Should be ~81%
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Molecular Property Prediction**: In PyG, load a molecular dataset (like `QM9`). Instead of Node Classification, do **Graph Classification**. Pass the atoms through 3 GCN layers, then use `global_mean_pool` to collapse all nodes into a single vector. Pass that vector through a Linear layer to predict the boiling point of the molecule!
- 🟡 **Link Prediction**: Build a GNN that predicts if two unconnected nodes *should* be connected (e.g., "Friend Recommendation"). Train it by randomly hiding 10% of the edges in a network, asking the GNN to output embeddings for all nodes, and calculating the dot product between node embeddings to predict edge existence.

### What's Next
| Next | Why |
|------|-----|
| [Self-Supervised Learning](./03-Self-Supervised-Learning.md) | We've looked at massive models, but they all require huge amounts of labeled data. What if you have 10 million images, but none of them have labels? We use Self-Supervised Learning (SSL). |

---

[← Deep Reinforcement Learning (DRL)](01-Reinforcement-Learning.md) | [Back to Index](../README.md) | [Next: Self-Supervised Learning →](03-Self-Supervised-Learning.md)
