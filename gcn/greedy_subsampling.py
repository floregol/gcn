import scipy.sparse as sp
from greedy_sampling.graph_generator import *
from greedy_sampling.linear_H import get_identity_H
from greedy_sampling.signal_generator import get_random_signal_zero_mean_circular
from greedy_sampling.sampling_algo import greedy_algo
from greedy_sampling.sampling_algo_util import *
import numpy as np
import scipy.sparse as sp
import time


def greedy_eigenvector_precomputation(adj, K_sparse, noise):
    now = time.time()
    dense_adj = sp.csr_matrix.todense(adj)
    num_nodes = dense_adj.shape[0]

    V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition_from_svd_adj(dense_adj, K_sparse)
    # Linear transformation of the signal
    H, H_h = get_identity_H(num_nodes)
    # Random signal and noise vectors
    x, cov_x = get_random_signal_zero_mean_circular(1.0, K_sparse)
    w, cov_w = get_random_signal_zero_mean_circular(noise, num_nodes)

    # Noisy observation. (Not used for now)
    #y = x + w
    print('time' + str(time.time() - now))
    return V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, num_nodes


def get_train_mask_greedy(label_percent, initial_train_mask, V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w,
                          num_nodes):

    train_index = np.argwhere(initial_train_mask).reshape(-1)
    #TODO filter by adj train
    train_mask = np.zeros((initial_train_mask.shape), dtype=bool)  # list of False

    random_sampling_set_size = int((label_percent / 100) * train_index.shape[0])
    # Get sampling set selected by the diff. algorithms
    greedy_subset = greedy_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, random_sampling_set_size,
                                num_nodes)
    train_mask[greedy_subset] = True
    return train_mask


# if __name__ == "__main__":
#     graph = generate_Erdos_Renyi_graph(13)
#     K_sparse = 3
#     adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()), weight='weight').toarray()
#     V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition_from_adj(adj, K_sparse)
#     V_ksparse_2, _, _ = get_sparse_eigen_decomposition_from_svd_adj(adj, K_sparse)
#     print(V_ksparse)
#     print(V_ksparse_2)