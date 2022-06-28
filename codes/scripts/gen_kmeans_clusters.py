import torch
import numpy as np
from random import shuffle
from tqdm import tqdm
import random
import os
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

use_cuda = False
dtype = torch.float32
device_id = 'cpu'


def load_vectors():
    """ Will need to be modified per-data type you are loading. """
    all_files = torch.load('/y/separated/large_mel_cheaters_linux.pth')
    os.makedirs('/y/separated/randomly_sampled_cheaters', exist_ok=True)
    vecs = []
    print("Gathering vectors..")
    j = 0
    for f in tqdm(all_files):
        vs=torch.tensor(np.load(f)['arr_0'])
        for k in range(4):
            vecs.append(vs[0,:,random.randint(0,vs.shape[-1]-1)])
        if len(vecs) >= 1000000:
            vecs = torch.stack(vecs, dim=0)
            torch.save(vecs, f'/y/separated/randomly_sampled_cheaters/{j}.pth')
            j += 1
            vecs = []
    vecs = [torch.stack(vecs, dim=0)]
    for i in range(j):
        vecs.append(torch.load(f'/y/separated/randomly_sampled_cheaters/{i}.pth'))
    vecs = torch.cat(vecs, dim=0)
    torch.save(vecs, '/y/separated/randomly_sampled_cheaters/combined.pth')

def k_means(x, K, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric.
       Thanks to https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    """

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in tqdm(range(Niter)):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c    


if __name__ == '__main__':
    #load_vectors()
    vecs = torch.load('/y/separated/randomly_sampled_cheaters/combined.pth')
    cl, c = k_means(vecs, 8192, 50)
    torch.save((cl, c), '/y/separated/randomly_sampled_cheaters/k_means_clusters.pth')
