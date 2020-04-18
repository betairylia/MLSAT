import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from sparsemax.sparsemax import Sparsemax

from torch_scatter import scatter

class MNISTEncoder(nn.Module): # network G

    def __init__(self, out_dim, in_ch = 1,  BN = True, small_model = False):
        super(MNISTEncoder, self).__init__()

        max_pool = 2 if small_model else 1

        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch * 2, 32, (5, 5), padding = 2),
            nn.BatchNorm2d(32) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(32, 64, (5, 5), padding = 2),
            nn.BatchNorm2d(64) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
        )

        # maybe not a good idea to use BN in dense part ... ?
        self.densenet = nn.Sequential(
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    
    def forward(self, x, y):
        
        x = torch.cat([x, y], dim = 1)
        n = self.convnet(x)
        n = self.densenet(n.view(n.shape[0], -1))

        return n

class MNISTEncoderBig(nn.Module): # network G

    def __init__(self, out_dim, in_ch = 1,  BN = True, small_model = False):
        super(MNISTEncoderBig, self).__init__()

        max_pool = 2 if small_model else 1

        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch * 2, 128, (5, 5), padding = 2),
            nn.BatchNorm2d(128) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(128, 256, (5, 5), padding = 2),
            nn.BatchNorm2d(256) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
        )

        # maybe not a good idea to use BN in dense part ... ?
        self.densenet = nn.Sequential(
            nn.Linear(12544, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    
    def forward(self, x, y):
        
        x = torch.cat([x, y], dim = 1)
        n = self.convnet(x)
        n = self.densenet(n.view(n.shape[0], -1))

        return n

class MNISTEncoderDotP(nn.Module): # network G

    def __init__(self, out_dim, in_ch = 1,  BN = True, small_model = False):
        super(MNISTEncoderDotP, self).__init__()

        max_pool = 2 if small_model else 1

        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch, 128, (5, 5), padding = 2),
            nn.BatchNorm2d(128) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(128, 256, (5, 5), padding = 2),
            nn.BatchNorm2d(256) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
        )

        # maybe not a good idea to use BN in dense part ... ?
        self.densenet = nn.Sequential(
            nn.Linear(12544, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    
    def forward(self, x, y):
        
        nx = self.convnet(x)
        nx = self.densenet(nx.view(nx.shape[0], -1))

        ny = self.convnet(y)
        ny = self.densenet(ny.view(ny.shape[0], -1))

        dot = torch.sum(nx * ny, dim = -1)

        return dot

class CIFAREncoderBig(nn.Module): # network G

    def __init__(self, out_dim, in_ch = 1,  BN = True, small_model = False):
        super(CIFAREncoderBig, self).__init__()

        max_pool = 2 if small_model else 1

        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch * 2, 128, (5, 5), padding = 2),
            nn.BatchNorm2d(128) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(128, 256, (5, 5), padding = 2),
            nn.BatchNorm2d(256) if BN else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
        )

        # maybe not a good idea to use BN in dense part ... ?
        self.densenet = nn.Sequential(
            nn.Linear(16384, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
    
    def forward(self, x, y):
        
        x = torch.cat([x, y], dim = 1)
        n = self.convnet(x)
        n = self.densenet(n.view(n.shape[0], -1))

        return n

class ML_SAT_ManifoldFit(nn.Module):

    def __init__(self, in_dim = 512, genlin_lambda = 0):
        
        super(ML_SAT_ManifoldFit, self).__init__()
        
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 1)
        # )
        self.net = MNISTEncoder(1, 1, False, True)
        # self.net = MNISTEncoderBig(1, 1, True, True)
        # self.net = CIFAREncoderBig(1, 3, True, True)
        # self.net = MNISTEncoderDotP(1, 1, False, True)

        # => [bs, 1, K]
        # => [bs, K]

        self.sparsemax = Sparsemax(dim = 1)
        self.genlin_lambda = genlin_lambda
        # self.sparsemax = nn.Softmax(dim = 1)

    def forward(self, x, y):
        
        K = x.shape[1]
        bs = x.shape[0]
        res = x.shape[2:]
        x = x.reshape(K*bs, *res)
        y = y.reshape(K*bs, *res)

        n = self.net(x, y)
        n = n.view(bs, K)

        n = self.sparsemax((1.0 / (1.0 - self.genlin_lambda)) * n)

        return n

class IdentityEmebdder(nn.Module):

    def __init__(self, in_dim = 784):

        super(IdentityEmebdder, self).__init__()
        self.out_dim = in_dim

    def forward(self, data):

        return data

class ML_SAT_model(nn.Module):

    def __init__(self, embedder, in_dim = 784, K = 15, genlin_lambda = 0):

        super(ML_SAT_model, self).__init__()

        self.K = K

        if embedder is None:
            embedder = IdentityEmebdder(in_dim = in_dim)
        
        self.embedder = embedder
        self.manifoldFit = ML_SAT_ManifoldFit(in_dim = 2 * self.embedder.out_dim, genlin_lambda = genlin_lambda)

    def forward(self, data): # Uses PyG

        data = self.embedder(data)

        x = data.x
        scatter_idx = data.scatter_idx
        bs = data.num_graphs
        batch = data.batch
        idxs = torch.stack([batch, scatter_idx], dim = 1)
        one_hop = torch.zeros((bs, self.K + 2, *x.shape[1:])).to(x.device)

        one_hop[idxs[:, 0], idxs[:, 1], :] += x # will this work? lol
        center = one_hop[:, 0, :].unsqueeze(1)
        one_hop = one_hop[:, 1:-1, :] # remove not one-hop entries

        # edges = torch.cat([center.repeat(1, self.K, *([1] * (len(center.shape) - 2))), one_hop], dim = 2) # concat to x0-xi
        # edges = edges.permute(0, 2, 1).contiguous()

        eweight = self.manifoldFit(center.repeat(1, self.K, *([1] * (len(center.shape) - 2))), one_hop)
        pred = torch.sum(eweight.unsqueeze(-1) * one_hop.view(bs, self.K, -1), 1)
        pred = pred.view(bs, *x.shape[1:])

        return eweight, pred, center, one_hop

def weight_reset(m):
    if callable(getattr(m, 'reset_parameters', None)):
        m.reset_parameters()
