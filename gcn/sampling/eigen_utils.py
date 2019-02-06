import scipy.sparse as sp
from greedy_sampling.graph_generator import *
from greedy_sampling.linear_H import get_identity_H
from greedy_sampling.signal_generator import get_random_signal_zero_mean_circular
from greedy_sampling.sampling_algo_util import *
import numpy as np
import time


def eigenvector_precomputation(adj, K_sparse, noise, initial_train_mask):

    train_index = np.argwhere(initial_train_mask).reshape(-1)
    now = time.time()
    dense_train_adj = sp.csr_matrix.todense(adj)
    num_nodes = dense_train_adj.shape[0]

    V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition_from_svd_adj(
        dense_train_adj, K_sparse)
    # Linear transformation of the signal
    H, H_h = get_identity_H(num_nodes)
    # Random signal and noise vectors
    x, cov_x = get_random_signal_zero_mean_circular(1.0, K_sparse)
    w, cov_w = get_random_signal_zero_mean_circular(noise, num_nodes)
    W = get_W(V_ksparse_H, H_h, H, V_ksparse)
    print('time' + str(time.time() - now))
    return V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, num_nodes

# Compute P = (L-aI)-^(-1)
def get_P_matrix(adj, lam):
    A = sp.csr_matrix.todense(adj)
    degree = np.array(np.sum(A, axis=0)[0])
    D = degree * np.eye(degree.shape[1])
    L = D - A
    return np.linalg.inv(L - lam * np.eye(degree.shape[1]))