import torch

# The graph Laplacian is defined assert create_graph_lapl(a).shape == (5, 5)
a = torch.rand(5, 5)
a[a > 0.5] = 1
a[a <= 0.5] = 0


def calc_degree_matrix(a):
    return torch.diag(a.sum(dim=-1))


def create_graph_lapl(a):
    return calc_degree_matrix(a) - a


print("A:", a)
print("L:", create_graph_lapl(a))


# rand binary Adj matrix
a = torch.rand(5, 5)
a[a > 0.5] = 1
a[a <= 0.5] = 0


def calc_degree_matrix_norm(a):
    return torch.diag(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.ones(size) - (D_norm @ a @ D_norm)
    return L_norm


print("A: ", a)
print("L_norm: ", create_graph_lapl_norm(a))

# In graphs, the smallest non-zero eigenvalue has been used for “spectral” image segmentation from the 90s. By converting a grayscale image to a graph (see code below), you can divide an image based on its slowest non-zero frequencies/eigenvalues.

import numpy as np
from scipy import misc
from skimage.transform import resize
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering


re_size = 64  # ownsampling of resized rectangular image
img = misc.face(gray=True)  # retrieve a grayscale image
img = resize(img, (re_size, re_size))
mask = img.astype(bool)
graph = img_to_graph(img, mask=mask)
# Take a decreasing function of the gradient: we take it weakly
# dependant from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())
labels = spectral_clustering(graph, n_clusters=3)
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.figure(figsize=(6, 3))
plt.imshow(img, cmap="gray", interpolation="nearest")

plt.figure(figsize=(6, 3))
plt.imshow(label_im, cmap=plt.cm.nipy_spectral, interpolation="nearest")
plt.show()

# COO Cordinate format
import numpy as np
import scipy.sparse as sparse

row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])

mtx = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
mtx.todense()

# ---------------------------------------------------------------------------------
# Math behind the GNN
import torch
import torch.nn as nn


def create_adj(size):
    a = torch.rand(size, size)
    a[a > 0.5] = 1
    a[a <= 0.5] = 0

    # for illustration we set the diagonal elemtns to zero
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i, j] = 0
    return a


def calc_degree_matrix(a):
    return torch.diag(a.sum(dim=-1))


def create_graph_lapl(a):
    return calc_degree_matrix(a) - a


def calc_degree_matrix_norm(a):
    return torch.diag(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.ones(size) - (D_norm @ a @ D_norm)
    return L_norm


def find_eigmax(L):
    with torch.no_grad():
        e1, _ = torch.eig(L, eigenvectors=False)
        return torch.max(e1[:, 0]).item()


def chebyshev_Lapl(X, Lapl, thetas, order):
    list_powers = []
    nodes = Lapl.shape[0]

    T0 = X.float()

    eigmax = find_eigmax(Lapl)
    L_rescaled = (2 * Lapl / eigmax) - torch.eye(nodes)

    y = T0 * thetas[0]
    list_powers.append(y)
    T1 = torch.matmul(L_rescaled, T0)
    list_powers.append(T1 * thetas[1])

    # Computation of: T_k = 2*L_rescaled*T_k-1  -  T_k-2
    for k in range(2, order):
        T2 = 2 * torch.matmul(L_rescaled, T1) - T0
        list_powers.append((T2 * thetas[k]))
        T0, T1 = T1, T2
    y_out = torch.stack(list_powers, dim=-1)
    # the powers may be summed or concatenated. i use concatenation here
    y_out = y_out.view(nodes, -1)  # -1 = order* features_of_signal
    return y_out


features = 3
out_features = 50
a = create_adj(10)
L = create_graph_lapl_norm(a)
x = torch.rand(10, features)
power_order = 4  # p-hops
thetas = nn.Parameter(torch.rand(4))

out = chebyshev_Lapl(x, L, thetas, power_order)

print("cheb approx out powers concatenated:", out.shape)
# because we used concatenation  of the powers
# the out features will be power_order * features
linear = nn.Linear(4 * 3, out_features)

layer_out = linear(out)
print("Layers output:", layer_out.shape)

# ---------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchnet as tnt
import networkx as nx

import os
import sys

cwd = os.getcwd()
# add MUTAG data in the environment
sys.path.append(cwd + "/../MUTAG")


""" Download MUTAG dataset"""
""" Extra graph utils and data loading stuff"""


def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]


def get_graph_signal(nx_graph):
    d = dict((k, v) for k, v in nx_graph.nodes.items())
    x = []
    invd = {}
    j = 0
    for k, v in d.items():
        x.append(v["attr_dict"])
        invd[k] = j
        j = j + 1
    return np.array(x)


def load_data(path, ds_name, use_node_labels=True, max_node_label=10):
    node2graph = {}
    Gs = []
    data = []
    dataset_graph_indicator = f"{ds_name}_graph_indicator.txt"
    dataset_adj = f"{ds_name}_A.txt"
    dataset_node_labels = f"{ds_name}_node_labels.txt"
    dataset_graph_labels = f"{ds_name}_graph_labels.txt"

    path_graph_indicator = os.path.join(path, dataset_graph_indicator)
    path_adj = os.path.join(path, dataset_adj)
    path_node_lab = os.path.join(path, dataset_node_labels)
    path_labels = os.path.join(path, dataset_graph_labels)

    with open(path_graph_indicator, "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open(path_adj, "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])] - 1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open(path_node_lab, "r") as f:
            c = 1
            for line in f:
                node_label = indices_to_one_hot(int(line[:-1]), max_node_label)
                Gs[node2graph[c] - 1].add_node(c, attr_dict=node_label)
                c += 1

    labels = []
    with open(path_labels, "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    return list(zip(Gs, labels))


def create_loaders(dataset, batch_size, split_id, offset=-1):
    train_dataset = dataset[:split_id]
    val_dataset = dataset[split_id:]
    return to_pytorch_dataset(train_dataset, offset, batch_size), to_pytorch_dataset(
        val_dataset, offset, batch_size
    )


def to_pytorch_dataset(dataset, label_offset=0, batch_size=1):
    # graphs, labels = dataset
    list_set = []
    for graph, label in dataset:
        F, G = get_graph_signal(graph), nx.to_numpy_matrix(graph)
        numOfNodes = G.shape[0]
        F_tensor = torch.from_numpy(F).float()
        G_tensor = torch.from_numpy(G).float()

        # fix labels to zero-indexing
        if label == -1:
            label = 0

        label += label_offset

        list_set.append(tuple((F_tensor, G_tensor, label)))

    dataset_tnt = tnt.dataset.ListDataset(list_set)
    data_loader = torch.utils.data.DataLoader(
        dataset_tnt, shuffle=True, batch_size=batch_size
    )
    return data_loader


dataset = load_data(
    path="../MUTAG", ds_name="MUTAG", use_node_labels=True, max_node_label=7
)
train_dataset, val_dataset = create_loaders(
    dataset, batch_size=1, split_id=150, offset=0
)
print("Data are ready")


def device_as(x, y):
    return x.to(y.device)


# tensor operationa now support batched inputs
def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    a += device_as(torch.eye(size), a)
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.bmm(torch.bmm(D_norm, a), D_norm)
    return L_norm


class GCN_Layer(nn.Module):
    """
    A simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, X, A):
        """
        A: adjαcency matrix
        X: graph signal
        """
        L = create_graph_lapl_norm(A)
        x = self.linear(X)
        return torch.bmm(L, x)


criterion = torch.nn.CrossEntropyLoss()
device = "cpu"

print(f"Training on {device}")
model = GNN(in_features=7, hidden_dim=128, classes=2).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(train_loader):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        X, A, labels = data
        X, A, labels = X.to(device), A.to(device), labels.to(device)
        # Forward pass.
        out = model(X, A)
        # Compute the graph classification loss.
        loss = criterion(out, labels)
        # Calculate gradients.
        loss.backward()
        # Updates the models parameters
        optimizer.step()


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        X, A, labels = data
        # Forward pass.
        out = model(X, A)
        # Take the index of the class with the highest probability.
        pred = out.argmax(dim=1)
        # Compare with ground-truth labels.
        correct += int((pred == labels).sum())
    return correct / len(loader.dataset)


best_val = -1
for epoch in range(1, 241):
    train(train_dataset)
    train_acc = test(train_dataset)
    val_acc = test(val_dataset)
    if val_acc > best_val:
        best_val = val_acc
        epoch_best = epoch

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} || Best Val Score: {best_val:.4f} (Epoch {epoch_best:03d}) "
        )
