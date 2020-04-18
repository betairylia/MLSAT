import matplotlib
matplotlib.use('Agg') 

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from annoy import AnnoyIndex
from matplotlib import pyplot as plt

from torch_geometric.data import Data, DataLoader
import torch_geometric

import os
import networkx as nx
import sys

from dataset import *
from model import *

import umap
from scipy.sparse import csr_matrix, lil_matrix
from UniversalCLI.CLI import CLI
from UniversalCLI.Components import *

import math
import numpy as np

from functools import reduce
from matplotlib import cm
from dataset import cmdimshow

from sklearn.mixture import GaussianMixture
from unsup_eval import ACC, NMI

from utils import *

from datetime import datetime

# Init CLI
cli_width= 100
cli_epch = Styled(20, 'left', style = ('green', None, ['bold']))
cli_msgt = Styled(10, 'left', style = ('cyan', None, ['bold']))
cli_msgc = CLIComponent(cli_width - 30, 'right')
cli_pBar = ProgressBar(cli_width, tot = 100)
cli_loss = CLIComponent(cli_width)

cli_msgt("Message - ")

name = sys.argv[1] if len(sys.argv) >= 2 else 'Unnamed'

CLI.init([[cli_epch, cli_msgt, cli_msgc], [cli_pBar], [cli_loss]], width = cli_width, title = name)

def mainLoop(pthprefix, exid):

    knn_k = options['K']
    model.K = knn_k
    model.manifoldFit.genlin_lambda = options['genlin_lambda']
    dset = MLSAT_MNIST(K = knn_k, train = options['trainSet'])
    # dset = MLSAT_CIFAR(K = knn_k, train = options['trainSet'])
    loader = torch_geometric.data.DataLoader(dset, batch_size = 128, shuffle = True, num_workers = 32)
    if options['VAT'] is True:
        VAT_loader = torch_geometric.data.DataLoader(dset, batch_size = 128, shuffle = True, num_workers = 32)
    model.apply(weight_reset)

    # path = input('Please input a path to save the original A-kNN graph:\n')
    path = "%s/original%d.gexf" % (pthprefix, exid)
    dset.generateGraph(path)
    CLI.log("Original ANN graph saved as " + colored("%s" % path, 'yellow'))

    CLI.log(colored("Training begins", 'red', None, ['bold']))

    for i in range(6): # Epochs

        cli_epch('Epoch %d:' % i)
        cli_pBar(0, len(loader.dataset), 'Train')
        
        if options['VAT']:
            vbloader = iter(VAT_loader)

        model.train()
        for batch in loader:
            
            eweight, pred, center, one_hop = model(batch)
            loss = loss_func(batch.center, pred)

            # Regularizer
            L0_reg = options['lambda_L0'] * torch.mean(torch.pow(eweight + 1e-5, 0.1))
            MLSATloss = loss.detach().cpu().item()
            loss += L0_reg

            if options['VAT']:

                vbatch = next(vbloader)

                veweight, _, _, _ = model(vbatch)
                
                VAT_d = torch.tensor(np.random.randn(*vbatch.x.shape).astype('f'), requires_grad = True)
                VAT_d = autoNormalize(VAT_d)
                VAT_r = options['VAT_xi'] * VAT_d
                VAT_r = VAT_r.to(vbatch.x.device)
                VAT_r.retain_grad()

                vbatch.x += VAT_r

                veweight_p, _, _, _ = model(vbatch)

                # Clear gradient
                optimizer.zero_grad()

                VAT_kl = torch.sum(JSDivergence(veweight, veweight_p))
                VAT_kl.backward(retain_graph = True)
                VAT_g = VAT_r.grad
                VAT_r_vadv = options['VAT_eps'] * autoNormalize(VAT_g)

                vbatch.x += VAT_r_vadv - VAT_r

                veweight_pp, _, _, _ = model(vbatch)
                
                VAT_loss = torch.sum(JSDivergence(veweight.detach(), veweight_pp)) / vbatch.x.shape[0]
            
            else:

                VAT_loss = torch.zeros_like(loss)

            @RedirectWrapper(target_cli = CLI)
            def foo():
                print("Prediction")
                cmdimshow(pred[0, 0, :].detach().cpu().numpy())
                print("GT")
                cmdimshow(batch.center[0, 0, :].detach().cpu().numpy())
                print("Mean of one_hop")
                cmdimshow(torch.mean(one_hop[0, :, 0], dim = 0).detach().cpu().numpy())

            # Show previews
            # foo()

            VAT_loss = options['VAT_lambda'] * VAT_loss
            loss += VAT_loss

            cli_loss.setContent(colored("Loss = %8.6f %s" % (loss.detach().cpu().item(), colored("(ML-SAT %8.6f | L0 %8.6f | VAT %8.6f)" % (MLSATloss, L0_reg.detach().cpu().item(), VAT_loss.detach().cpu().item()), 'cyan')), 'green'), 62)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cli_pBar.updateProgressInc(batch.num_graphs)

            # CLI.render()

            # break

    CLI.log(colored("Generating Graph ...", 'cyan'))
    cli_epch("Graph Gen")
    cli_pBar(0, len(loader.dataset), "Graph")

    eps = 1e-5
    model.eval()
    G = nx.Graph()
    for i in range(len(dset.Y)):
        G.add_node(i, digit = dset.Y[i].item())

    nX = len(dset)
    NNMat = np.ones((nX, nX), dtype = np.float32) * math.sqrt((1.0 / eps))

    for batch in loader:
        eweight, pred, center, one_hop = model(batch)
        for i in range(len(batch.raw_edges)):
            # @RedirectWrapper(target_cli = CLI)
            # def foo():
            NNMat[batch.raw_edges[i, 0].item(), batch.raw_edges[i, 1].item()] = math.sqrt(math.log(1.0 / (max(eweight[i // knn_k, i % knn_k].item(), eps))))
            # NNMat[batch.raw_edges[i, 1].item(), batch.raw_edges[i, 0].item()] = math.sqrt(math.log(1.0 / (max(eweight[i // knn_k, i % knn_k].item(), eps))))
            # foo()
            if eweight[i // knn_k, i % knn_k].item() > eps:
                G.add_edge(batch.raw_edges[i, 0].item(), batch.raw_edges[i, 1].item(), weight = eweight[i // knn_k, i % knn_k].item())
        cli_pBar.updateProgressInc(batch.num_graphs)

    # NNMat = csr_matrix(NNMat)
    CLI.log(colored("Graph - Vcount %6d; Ecount %6d." % (len(dset.Y), len(G.edges())), 'cyan'))

    cli_msgc("Saving Graph to disk")
    # path = input('Please input a path to save the ML_SAT graph:\n')
    path = "%s/ML_SAT%d.gexf" % (pthprefix, exid)
    # path = "%s_ML_SAT.gexf" % name
    nx.write_gexf(G, path)

    CLI.log("Graph saved as " + colored("%s" % path, 'yellow'))

    CLI.log(colored("UMAP begins", 'red', None, ['bold']))

    @RedirectWrapper(target_cli = CLI)
    def get_emb():
        reducer = umap.UMAP(n_neighbors = knn_k, n_components = 2, metric = "precomputed", verbose = False, min_dist = options['min_dist'])
        emb = reducer.fit_transform(NNMat)
        return emb

    CLI.log(colored("UMAP 2 begins", 'red', None, ['bold']))

    @RedirectWrapper(target_cli = CLI)
    def get_emb_orig():
        reducer = umap.UMAP(n_neighbors = knn_k, n_components = 2, verbose = False, min_dist = options['min_dist'])
        emb = reducer.fit_transform(dset.X.reshape(nX, -1))
        return emb

    emb = get_emb()
    emb_orig = get_emb_orig()
    np.save("%s/umap_MLSAT%d.npy" % (pthprefix, exid), emb)
    np.save("%s/umap%d.npy" % (pthprefix, exid), emb_orig)

    CLI.log("Embeddings saved as " + colored("umap.npy, umap_MLSAT.npy, X.npy & Y.npy", 'yellow', None, ['bold']))

    return emb, emb_orig, dset.Y

if __name__ == "__main__":

    options =\
    {
        'K': 15,
        'min_dist': 0.1,
        'lambda_L0': 1,
        'trainSet': False,
        # 'trainSet': True,
        'VAT': True,
        'VAT_xi': 10,
        'VAT_eps': 1,
        'VAT_lambda': 25,
        'genlin_lambda': 0,
    }

    hpgrid =\
    {
        # 'K': [15],
        # 'min_dist': [0.0],
        'lambda_L0': [0, 1],
        'genlin_lambda': [0, 0.5],
        'VAT_lambda': [1, 10, 25],

        # 'K': [15],
        # 'min_dist': [0.0],
        # 'lambda_L0': [1.0],
        # 'genlin_lambda': [0, 0.5, 0.8],
    }
    
    hpcount = reduce(lambda x, y: x * y, [len(hpgrid[key]) for key in hpgrid.keys()], 1)

    model = ML_SAT_model(None, 1024, 20, options['genlin_lambda'])
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    ######

    finish = False

    cnt = 0

    pthprefix = "Results/%s%s/" % (name, datetime.now().strftime("[%Y-%m-%d-%H-%M-%S]"))
    os.makedirs(pthprefix)

    result_log = open("%s/result.txt" % pthprefix, 'w')

    hpstats = {}
    for key in hpgrid:
        hpstats[key] = 0
        options[key] = hpgrid[key][hpstats[key]]

    while not finish:
        # Generate desc string
        this_str = ''
        for key in hpgrid:
            this_str += "%s = %s; " % (key, repr(hpgrid[key][hpstats[key]]))

        cnt += 1

        cli_msgc(colored(this_str, 'magenta'), len(this_str))
        cli_msgt("HP %2d / %2d" % (cnt, hpcount))

        this_str = "HP = {%s}" % this_str
        CLI.log(colored("START: ", 'green') + colored(this_str, 'cyan'))
        
        result_log.write("\n =*=*=*=*=*=*=*=*=*=*=\n")
        result_log.write(this_str + "\n")

        emb, emb_o, Y = mainLoop(pthprefix, cnt)

        # Perform GMMC
        def gmmc(savename, temb):

            gmm = GaussianMixture(n_components = 10)
            GMMC_pred = gmm.fit_predict(temb)

            score_ACC = ACC(10, Y, GMMC_pred)
            score_NMI = NMI(Y, GMMC_pred)

            _X, _Y = np.meshgrid(np.linspace(temb[:,0].min(), temb[:,0].max()), np.linspace(temb[:,1].min(), temb[:,1].max()))
            XX = np.array([_X.ravel(), _Y.ravel()]).T
            Z = gmm.score_samples(XX)
            Z = Z.reshape((50,50))

            new_str = ("(%s)" % savename) + this_str + "ACC = %8.6f, NMI = %8.6f" % (score_ACC, score_NMI)

            # Generate UMAP img
            colormap = cm.get_cmap('tab10')
            plt.figure(figsize=(16, 12))
            vis = plt.scatter(temb[:, 0], temb[:, 1], c = Y, cmap = colormap, alpha = 0.15)
            plt.contour(_X, _Y, Z)
            plt.colorbar(vis)
            plt.title(new_str)
            plt.savefig("%s/%s_HP%d.png" % (pthprefix, savename, cnt))

            result_log.write("%10s: ACC = %8.6f, NMI = %8.6f" % (savename, score_ACC, score_NMI) + "\n")
            result_log.flush()

        gmmc("ML-SAT", emb)
        gmmc("Original", emb_o)

        # update HP
        finish = True
        for key in hpgrid:
            hpstats[key] += 1
            hpstats[key] = hpstats[key] % len(hpgrid[key])
            if hpstats[key] != 0:
                finish = False
                break
    
    result_log.close()
