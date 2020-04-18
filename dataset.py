import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from annoy import AnnoyIndex
from matplotlib import pyplot as plt

from torch_geometric.data import Data

import os
import networkx as nx

from UniversalCLI.CLI import CLI
from UniversalCLI.Components import *

import numpy as np

class MLSAT_ANNSet(Dataset):

    @RedirectWrapper(target_cli = CLI)
    def __init__(self, K = 15, train = False):

        self.load(train = train)

        # Run Approximate NN search

        self.annoy = AnnoyIndex(self.dim, 'euclidean') # 'angular' ?

        if os.path.exists(self.annpth):
            self.annoy.load(self.annpth)
            print("Loaded ANN indices from %s" % self.annpth)
        
        else:
            print("Creating ANN indices ...")

            self.X = self.X.view(-1, self.dim) # 28 * 28 = 784

            for i, x in enumerate(self.X):
                self.annoy.add_item(i, x)
            
            self.annoy.build(128)
            self.annoy.save(self.annpth)

            print("ANN index complete")
        
        self.size = len(self.X)
        self.X = self.X.view(-1, *self.shape).contiguous().numpy()
        # self.X = self.X.view(-1, 784).contiguous().numpy()
        self.K = K

        self.ANNIdx = np.zeros((len(self.Y), self.K)).astype(np.int32)
        for i in range(len(self.Y)):
            self.ANNIdx[i] = self.annoy.get_nns_by_item(i, self.K+1)[1:]

    def load(self, train = False):
        raise NotImplementedError

    def __getitem__(self, idx):

        # No GCN now
        # neighbors = self.annoy.get_nns_by_item(idx, self.K+1)[1:]
        neighbors = self.ANNIdx[idx]
        edge_index = torch.tensor([[0] * self.K, [i for i in range(1, self.K+1)]], dtype = torch.long) # 1-hop neighbor
        x = torch.tensor([self.X[idx]] + [self.X[i] for i in neighbors], dtype = torch.float)
        
        sid = []
        for i in range(len(x)):
            sid.append(i if i <= self.K else (self.K + 1))
        
        scatter_idx = torch.tensor(sid, dtype = torch.long)
        raw_edges = torch.tensor([[idx, i] for i in neighbors], dtype = torch.long)
        center = torch.tensor([self.X[idx]], dtype = torch.float)

        data = Data(x = x, edge_index = edge_index, scatter_idx = scatter_idx, raw_edges = raw_edges, center = center)

        return data

    def generateGraph(self, path):

        G = nx.Graph()
        for i, x in enumerate(self.X):
            G.add_node(i, digit = self.Y[i].item())
        for i in range(len(self.X)):
            for j in self.annoy.get_nns_by_item(i, self.K+1)[1:]:
                G.add_edge(i, j)

        nx.write_gexf(G, path)

    def __len__(self):
        return self.size

class MLSAT_MNIST(MLSAT_ANNSet):

    @RedirectWrapper(target_cli = CLI)
    def load(self, train = False):

        print("Loading MNIST ...")

        # Get MNIST
        mnist_trainset = datasets.MNIST(root = './data', train = train, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        full_loader = DataLoader(mnist_trainset, batch_size = len(mnist_trainset), num_workers = 10, shuffle = False)
        
        self.X, self.Y = next(iter(full_loader))
        self.annpth = './data/MNIST/AnnoyIndex%s.ann' % ('train' if train else '')
        self.dim = 784
        self.shape = (1, 28, 28)

        del full_loader
        del mnist_trainset

class MLSAT_CIFAR(MLSAT_ANNSet):

    @RedirectWrapper(target_cli = CLI)
    def load(self, train = False):

        print("Loading CIFAR ...")

        # Get CIFAR
        cifar_trainset = datasets.CIFAR10(root = './data', train = train, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        full_loader = DataLoader(cifar_trainset, batch_size = len(cifar_trainset), num_workers = 10, shuffle = False)
        
        self.X, self.Y = next(iter(full_loader))
        self.annpth = './data/CIFAR10/AnnoyIndex%s.ann' % ('train' if train else '')
        self.dim = 3072
        self.shape = (3, 32, 32)

        del full_loader
        del cifar_trainset

def cmdimshow(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print("%s" % ('*' if arr[i, j] > 1e-3 else ' '), end = '')
        print('')

if __name__ == '__main__':

    dset = MLSAT_MNIST()
    print(dset.X.shape, dset.Y.shape)
    
    root = 102

    a = dset.annoy.get_nns_by_item(root, 10)
    cmdimshow(dset.X[root].view(28, 28))
    for i in a:
        print("%d [Digit: %d]" % (i, dset.Y[i]))
        cmdimshow(dset.X[i].view(28, 28))

    path = input('Please input a path to save the graph:\n')
    dset.generateGraph(path)
