from sampling.greedy_subsampling import *
from sampling.sampler import Sampler
from sampling.eigen_utils import eigenvector_precomputation
from datetime import datetime
from tqdm import tqdm
import scipy.sparse as sp
from greedy_sampling.graph_generator import *
from greedy_sampling.sampling_algo import greedy_algo
from greedy_sampling.sampling_algo_util import *
import numpy as np

import random


class GreedySampler(Sampler):
    def __init__(self, initial_train_mask, adj, K_sparse_list, noise_list):
        self.name = "Greedy"
        self.K_sparse_list = K_sparse_list
        self.K_sparse = 0
        self.noise_list = noise_list
        self.noise = 0
        self.multi_trials = False
        super(GreedySampler, self).__init__(initial_train_mask, adj)

    def precomputations(self):
        self.V_ksparse, self.V_ksparse_H, _, self.H, self.H_h, self.cov_x, self.cov_w, self.W, self.num_nodes = eigenvector_precomputation(
            self.adj, self.K_sparse, self.noise, self.initial_train_mask)
        greedy_subset = greedy_algo(
                self.V_ksparse, self.V_ksparse_H, self.get_v, self.H, self.H_h,
                self.cov_x, self.cov_w, self.W, self.initial_train_mask.shape[0],
                self.num_nodes, True, False)
        print(greedy_subset[0:10])
        self.greedy_training_nodes = []
        for node in greedy_subset:
            if node in self.train_index:
                self.greedy_training_nodes.append(node)
        print(self.greedy_training_nodes[0:10])
    def next_parameter(self):
        if (self.sampling_config_index == len(self.noise_list) * len(
                self.K_sparse_list)):
            return False
        self.K_sparse = self.K_sparse_list[self.sampling_config_index % len(
            self.noise_list)]
        self.noise = self.noise_list[int(
            self.sampling_config_index / len(self.noise_list))]
        print(str(self.K_sparse) + "," + str(self.noise))
        self.sampling_config_index += 1
        return True

    def get_results_tuple(self, fileinfo, settings, result):
        super(GreedySampler, self).set_info(settings, result)
        self.info['K_sparsity'] = self.K_sparse
        self.info['noise'] = self.noise
        results_filename = fileinfo + 'K_sparsity=' + str(
            self.K_sparse) + '_label_balance=Greedy_noise=' + str(
                self.noise) + '_results.p'
        return (self.dict_output, results_filename)

    def get_v(self, index):
        v_index = self.V_ksparse_H[:, index]
        v_index_H = self.V_ksparse[index, :]
        return v_index, v_index_H

    def get_train_mask_fun(self, seed):
        #TODO filter by adj train
        train_mask = np.zeros(
            (self.initial_train_mask.shape), dtype=bool)  # list of False
        if self.label_percent == 100:
            greedy_subset = self.train_index
        else:
            random_sampling_set_size = int((self.label_percent / 100) * self.train_index.shape[0])
            greedy_subset = self.greedy_training_nodes[0:random_sampling_set_size]
                
            # Get sampling set selected by the diff. algorithms
        print(greedy_subset[0:10])
        #greedy_subset = random.sample(range(train_index.shape[0]), random_sampling_set_size)
        train_mask[greedy_subset] = True
        mask = np.ones((self.initial_train_mask.shape), dtype=bool)
        mask[self.train_index] = 0
        train_mask[mask] = False
        label_percent = (100 * np.sum(train_mask) / self.train_index.shape[0])
        return train_mask, label_percent
