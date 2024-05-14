import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

def load_data(batch_size=64, train=True):
    dataset = MNISTSuperpixels(root='/tmp/MNISTSuperpixels', train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
