#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
from random import shuffle

import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--xfile", type=str, default="mnist2500_X.txt", help="file name of feature stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")

opt = parser.parse_args()
print("get choice from args", opt)
xfile = opt.xfile

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims).to('cuda')  # Sending to('cuda') after because torch.eig is broken in Windows currently on Ampere GPUs.
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in tqdm(range(max_iter)):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


def run_tsne_instance_level():
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    limit = 4000
    X, files = torch.load('../results_instance_resnet.pth')
    zipped = list(zip(X, files))
    shuffle(zipped)
    X, files = zip(*zipped)
    X = torch.cat(X, dim=0).squeeze()[:limit]
    labels = np.zeros(X.shape[0])  # We don't have any labels..

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 2048, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")

    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()
    torch.save((Y, files[:limit]), "../tsne_output.pth")


# Uses the results from the calculation above to create a **massive** pdf plot that shows 1/8 size images on the tsne
# spectrum.
def plot_instance_level_results_as_image_graph():
    Y, files = torch.load('../tsne_output.pth')
    fig, ax = pyplot.subplots()
    fig.set_size_inches(200,200,forward=True)
    ax.update_datalim(np.column_stack([Y[:, 0], Y[:, 1]]))
    ax.autoscale()

    for b in tqdm(range(Y.shape[0])):
        im = pyplot.imread(files[b])
        im = OffsetImage(im, zoom=1/2)
        ab = AnnotationBbox(im, (Y[b, 0], Y[b, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.scatter(Y[:, 0], Y[:, 1])

    pyplot.savefig('tsne.pdf')


random_coords = [(8,8),(12,12),(18,18),(24,24)]
def run_tsne_pixel_level():
    limit = 4000

    '''  # For spinenet-style latent dicts
    latent_dict = torch.load('../results/byol_latents/latent_dict_1.pth')
    id_vals = list(latent_dict.items())
    ids, X = zip(*id_vals)
    X = torch.stack(X, dim=0)[:limit//4]
    # Unravel X into 4 latents per image, chosen from fixed points. This will serve as a psuedorandom source since these
    # images are not aligned.
    b,c,h,w = X.shape
    X_c = []
    for rc in random_coords:
        X_c.append(X[:, :, rc[0], rc[1]])
    X = torch.cat(X_c, dim=0)
    '''

    # For resnet-style latent tuples
    X, files = torch.load('../../results/2021-4-8-imgset-latent-dict.pth')
    zipped = list(zip(X, files))
    shuffle(zipped)
    X, files = zip(*zipped)

    X = torch.stack(X, dim=0)[:limit//4]
    # Unravel X into 1 latents per image, chosen from fixed points. This will serve as a psuedorandom source since these
    # images are not aligned.
    X_c = []
    for rc in random_coords:
        X_c.append(X[:, 0, :, rc[0], rc[1]])
    X = torch.cat(X_c, dim=0)

    labels = np.zeros(X.shape[0])  # We don't have any labels..

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 128, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")

    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()
    torch.save((Y, files[:limit//4]), "../tsne_output_pix.pth")


# Uses the results from the calculation above to create a **massive** pdf plot that shows 1/8 size images on the tsne
# spectrum.
def plot_pixel_level_results_as_image_graph():
    Y, files = torch.load('../tsne_output_pix.pth')
    fig, ax = pyplot.subplots()
    fig.set_size_inches(200,200,forward=True)
    ax.update_datalim(np.column_stack([Y[:, 0], Y[:, 1]]))
    ax.autoscale()

    expansion = 8  # Should be latent_compression(=8) * image_compression_at_inference(=1)
    margins = 4  # Keep in mind this will be multiplied by <expansion>
    for b in tqdm(range(Y.shape[0])):
        if b % 4 == 0:
            id = b // 4
            imgfile = files[id]
            baseim = pyplot.imread(imgfile)

        ct, cl = random_coords[b%4]
        im = baseim[expansion*(ct-margins):expansion*(ct+margins),
                    expansion*(cl-margins):expansion*(cl+margins),:]
        im = OffsetImage(im, zoom=1)
        ab = AnnotationBbox(im, (Y[b, 0], Y[b, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.scatter(Y[:, 0], Y[:, 1])

    pyplot.savefig('tsne_pix.pdf')


def run_tsne_segformer():
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    limit = 10000
    X, points, files = torch.load('../results_segformer.pth')
    zipped = list(zip(X, points, files))
    shuffle(zipped)
    X, points, files = zip(*zipped)
    X = torch.cat(X, dim=0).squeeze()[:limit]
    labels = np.zeros(X.shape[0])  # We don't have any labels..

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 1024, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")

    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()
    torch.save((Y, points, files[:limit]), "../tsne_output.pth")


# Uses the results from the calculation above to create a **massive** pdf plot that shows 1/8 size images on the tsne
# spectrum.
def plot_segformer_results_as_image_graph():
    Y, points, files = torch.load('../tsne_output.pth')
    fig, ax = pyplot.subplots()
    fig.set_size_inches(200,200,forward=True)
    ax.update_datalim(np.column_stack([Y[:, 0], Y[:, 1]]))
    ax.autoscale()

    margins = 32
    for b in tqdm(range(Y.shape[0])):
        imgfile = files[b]
        baseim = pyplot.imread(imgfile)
        ct, cl = points[b]

        im = baseim[(ct-margins):(ct+margins),
                    (cl-margins):(cl+margins),:]
        im = OffsetImage(im, zoom=1)
        ab = AnnotationBbox(im, (Y[b, 0], Y[b, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.scatter(Y[:, 0], Y[:, 1])

    pyplot.savefig('tsne_segformer.pdf')


if __name__ == "__main__":
    # For use with instance-level results (e.g. from byol_resnet_playground.py)
    #run_tsne_instance_level()
    #plot_instance_level_results_as_image_graph()

    # For use with pixel-level results (e.g. from byol_uresnet_playground)
    #run_tsne_pixel_level()
    #plot_pixel_level_results_as_image_graph()

    # For use with segformer results
    run_tsne_segformer()
    #plot_segformer_results_as_image_graph()