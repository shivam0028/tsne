# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:58:01 2019

@author: tripa
"""

import numpy as np
from utils import normalize
from sklearn.datasets import make_swiss_roll

import matplotlib.pyplot as plt


def p_matrix(X=np.array([]), sigma=5.0):

    # Initialize some variables
    (n, d) = X.shape
    # Better way to calculate ||Xi - Xj||^2 for all pairs
    sum_X = np.sum(np.square(X), 1)
    D = - np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    D = D / (2 * sigma ** 2)
    P = np.exp(D)
    P[range(n), range(n)] = 0
    # Divide each element by sum of its row
    P = P/P.sum(axis=1)[:, None]

    return P


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances.")
    (n, d) = X.shape

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA.")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def sne(X=np.array([]), no_dims=2, initial_dims=50, max_iter=1000,
        sigma=5.0, perplexity=None, use_perplexity=False):
    """
        Runs SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    eta = 0.1
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))

    # Compute P-values based on perplexity by using
    # binary search or by using specified sigma
    if use_perplexity:
        P = x2p(X, 1e-5, perplexity)
        P[range(n), range(n)] = 0
        # Divide each element by sum of its row
        P = P/P.sum(axis=1)[:, None]
    else:
        P = p_matrix(X, sigma)

    P = np.maximum(P, 1e-12)

    # Variable to store cost at each run
    C = np.zeros((max_iter))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        D = - np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
        Q = np.exp(D)
        Q[range(n), range(n)] = 0
        # Divide each element by sum of its row
        Q = Q/Q.sum(axis=1)[:, None]
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P + P.T - Q - Q.T

        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T
                              * (Y[i, :] - Y), 0)

        # Perform the update
        Y = Y - eta * dY

        C[iter] = np.sum(P * np.log(P / Q))

        # Display result at 100 steps
        if (iter + 1) % 10 == 0:
            print("Iteration %d: error is %f" % (iter + 1, C[iter]))

        # Break if the convergence has became slow
        if iter > 1 and C[iter-1] - C[iter] < 0.001:
            print("Updating eta to ", eta/10)
            eta /= 10
            if eta < 0.001:
                print("Not converging much.")
                break

    # Return solution and the cost at each iteration
    return Y, C
# %%


dataset = 'swissroll'
X, labels = make_swiss_roll(n_samples=500)
X = normalize(X)

# Run algorithm with different values of sigma
for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    print("Running on Sigma = ", sigma)
    Y, C = sne(X, no_dims=2, initial_dims=50, max_iter=5000, sigma=sigma)

    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.savefig('sne_'+dataset+'_sigma_'+str(sigma)+'.eps',
                format='eps', dpi=1000)
    plt.close()
    np.savetxt('sne_'+dataset+'_sigma_cost_'+str(sigma)+'.txt', C)

# Run algorithm with different values of perplexity
for perplexity in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]:
    print("Running on Perplexity = ", perplexity)
    Y, C = sne(X, no_dims=2, initial_dims=50, max_iter=5000,
               perplexity=perplexity, use_perplexity=True)
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.savefig('sne_'+dataset+'_perplexity_'+str(perplexity)+'.eps',
                format='eps', dpi=1000)
    plt.close()
    np.savetxt('sne_'+dataset+'_perplexity_cost_'+str(perplexity)+'.txt', C)
