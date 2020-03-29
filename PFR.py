"""
Implementation of the VLDB 2019 paper
Operationalizing Individual Fairness with Pairwise Fair Representations
URL: https://dl.acm.org/doi/abs/10.14778/3372716.3372723
citation:
@article{10.14778/3372716.3372723,
	author = {Lahoti, Preethi and Gummadi, Krishna P. and Weikum, Gerhard},
	title = {Operationalizing Individual Fairness with Pairwise Fair Representations},
	year = {2019},
	issue_date = {December 2019},
	publisher = {VLDB Endowment},
	volume = {13},
	number = {4},
	issn = {2150-8097},
	url = {https://doi.org/10.14778/3372716.3372723},
	doi = {10.14778/3372716.3372723},
	journal = {Proc. VLDB Endow.},
	month = dec,
	pages = {506–518},
	numpages = {13}
}

__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""
from __future__ import division
import numpy as np
from scipy.linalg import eigh
from  scipy.sparse import csgraph

class PFR:
    def __init__(self, k, W_s, W_F, gamma = 1.0, exp_id='', alpha = None, normed = False):
        """
        Initializes the model.

        :param k:       Hyperparam representing the number of latent dimensions.
        :param W_s:     The adjacency matrix of k-nearest neighbour graph over input space X
        :param W_F:     The adjacency matrix of the pairwise fairness graph G associated to the problem.
        :param nn_k:    Hyperparam that controls the number of neighbours considered in the similarity graph.
        :param gamma:   Hyperparam controlling the influence of W^F.
        :param alpha:   Hyperparam controlling the influence of W^X. If none, default is set to 1 - gamma.
        """
        self.k = k
        self.W_F = W_F
        self.W_s = W_s
        self.exp_id = exp_id
        self.gamma = gamma
        if alpha != None:
            self.alpha = alpha
        else:
            self.alpha = 1 - self.gamma
        self.normed = normed

    def fit(self, X):
        """
        Learn the model using the training data.

        :param X:     Training data.
        """
        print('Just fitting')
        W = (self.alpha * self.W_s) + (self.gamma * self.W_F)
        L, diag_p = csgraph.laplacian(W, normed=self.normed, return_diag=True)

        # - Formulate the eigenproblem.
        lhs_matrix = (X.T.dot(L.dot(X)))
        rhs_matrix = None

        # - Solve the problem
        eigval, eigvec = eigh(a=lhs_matrix,
                                b=rhs_matrix,
                                overwrite_a=True,
                                overwrite_b=True,
                                check_finite=True)
        eigval = np.real(eigval)

        # - Select eigenvectors based on eigenvalues
        # -- get indices of k smallest eigen values
        k_eig_ixs = np.argpartition(eigval, self.k)[:self.k]

        # -- columns of eigvec are the eigen vectors corresponding to the eigen values
        # --- select column vectors corresponding to k largest eigen values
        self.V = eigvec[:, k_eig_ixs]

    def transform(self, X):
        return (self.V.T.dot(X.T)).T

    def project(self, X):
        Z = self.transform(X)
        proj = Z.dot(self.V.T)
        np.save('AFR_k{}_p{}_id{}_'.format(self.k, self.nn_k, self.exp_id), proj)
        return proj

    def fit_transform(self, X):
        """
        Learns the model from the training data and returns the data in the new space.

        :param X:   Training data.
        :return:    Training data in the new space.
        """
        print('Fitting and transforming...')
        self.fit(X)
        return self.transform(X)

