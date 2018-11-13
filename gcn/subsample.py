import random
import time
import numpy as np
import copy
from itertools import compress
import scipy.sparse as sp
from greedy_sampling.graph_generator import get_sparse_eigen_decomposition_from_adj
from greedy_sampling.linear_H import get_identity_H
from greedy_sampling.signal_generator import get_random_signal_zero_mean_circular
from greedy_sampling.sampling_algo import greedy_algo
from numpy.linalg import multi_dot
"""
Helper functions for sampling algorithms. 
"""


# W = V_k_H * H_H * H * V_K
def get_W(V_ksparse_H, H_h, H, V_ksparse):
    W = multi_dot([V_ksparse_H, H_h, H, V_ksparse])
    return W


random.seed(123)


#remove columns from adj matrix.
#TODO needs additional scaling?
#Be carefull too not modify the initial complete support matrix
def get_sub_sampled_support(complete_support, node_to_keep):
    index_array = complete_support[0][:]  # make a copy to avoid modifying complete support
    values = np.zeros(complete_support[1].shape)
    index_array_sorted = index_array[:, 1].argsort()
    j = 0
    node_to_keep.sort()
    for index_to_keep in node_to_keep:
        while (j < len(index_array_sorted) and index_to_keep >= index_array[index_array_sorted[j]][1]):
            if (index_to_keep == index_array[index_array_sorted[j]][1]):
                values[index_array_sorted[j]] = complete_support[1][index_array_sorted[j]]
            j += 1
    sub_sampled_support = (index_array, values, complete_support[2])

    return sub_sampled_support


# Return a train mask for label_percent of the trainig set.
# if maintain_label_balance, keep smallest number of labels per class in training set that respect the label_percent, except for 100 %
def get_train_mask(label_percent, y_train, initial_train_mask, adj, maintain_label_balance=False):

    train_index = np.argwhere(initial_train_mask).reshape(-1)
    train_mask = np.zeros((initial_train_mask.shape), dtype=bool)  # list of False
    if maintain_label_balance:
        ones_index = []
        for i in range(y_train.shape[1]):  # find the ones for each class
            ones_index.append(np.argwhere(y_train[train_index, i] > 0).reshape(-1))

        if label_percent < 100:
            smaller_num = min(
                int(len(l) * (label_percent / 100))
                for l in ones_index)  # find smaller number of ones per class that respect the % constraint
            print(smaller_num)
            for ones in ones_index:
                random_index = random.sample(list(ones), smaller_num)
                train_mask[random_index] = True  # set the same number of ones for each class, so the set is balanced
        else:
            for ones in ones_index:
                train_mask[ones] = True

    else:
        random_sampling_set_size = int((label_percent / 100) * train_index.shape[0])
        random_list = random.sample(range(train_index.shape[0]), random_sampling_set_size)
        train_mask[random_list] = True

    return train_mask


def get_train_mask_greedy(label_percent, y_train, initial_train_mask, adj, maintain_label_balance=False):
    now = time.time()
    dense_adj = sp.csr_matrix.todense(adj)
    K_sparse = 10
    noise = 0.01
    num_nodes = dense_adj.shape[0]

    train_index = np.argwhere(initial_train_mask).reshape(-1)
    #TODO filter by adj train
    train_mask = np.zeros((initial_train_mask.shape), dtype=bool)  # list of False

    random_sampling_set_size = int((label_percent / 100) * train_index.shape[0])
    print(random_sampling_set_size)
    V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition_from_adj(dense_adj, K_sparse)
    print('done')
    # Linear transformation of the signal
    H, H_h = get_identity_H(num_nodes)
    # Random signal and noise vectors
    x, cov_x = get_random_signal_zero_mean_circular(1.0, K_sparse)
    w, cov_w = get_random_signal_zero_mean_circular(noise, num_nodes)

    # Noisy observation. (Not used for now)
    #y = x + w

    # Pre computation
    W = get_W(V_ksparse_H, H_h, H, V_ksparse)

    # Get sampling set selected by the diff. algorithms
    greedy_subset = greedy_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, random_sampling_set_size,
                                num_nodes)
    print(greedy_subset)
    print('time' + str(time.time() - now))
    exit()
    return train_mask


#returns a random list of indexes of the node to be kept at random.
def get_random_percent(num_nodes, percent):
    if percent > 100:
        print("This is not how percentage works.")
        exit()
    random_sampling_set_size = int((percent * num_nodes) / 100)
    return random.sample(range(num_nodes), random_sampling_set_size)


#returns a list of indexes for the mask
def get_list_from_mask(mask):
    return list(compress(range(len(mask)), mask))


# Set features of node that shouldn't be in the set to crazy things to make sure they are not in the gcnn
def modify_features_that_shouldnt_change_anything(features, note_to_keep):
    note_doesnt_exist = [x for x in range(features[2][0]) if x not in note_to_keep]
    a = np.where(np.isin(features[0][:, 0], note_doesnt_exist))
    features[1][a[0]] = 10000000
