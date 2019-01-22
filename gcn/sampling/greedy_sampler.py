from sampling.greedy_subsampling import *
from sampling.sampler import Sampler
from datetime import datetime
from tqdm import tqdm


class GreedySampler(Sampler):
    def __init__(self, initial_train_mask, adj, K_sparse_list, noise_list):
        self.name = "Greedy"
        self.K_sparse_list = K_sparse_list
        self.K_sparse = 0
        self.noise_list = noise_list
        self.noise = 0
        super(GreedySampler, self).__init__(initial_train_mask, adj)

    def precomputations(self):
        self.V_ksparse, self.V_ksparse_H, self.get_v, self.H, self.H_h, self.cov_x, self.cov_w, self.W, self.num_nodes = greedy_eigenvector_precomputation(
            self.adj, self.K_sparse, self.noise, self.initial_train_mask)

    def next_parameter(self):
        if (self.sampling_config_index == len(self.noise_list) * len(
                self.K_sparse_list)):
            return False
        self.K_sparse = self.K_sparse_list[self.sampling_config_index %
                                           len(self.noise_list)]
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

    def get_train_mask_fun(self, seed):
        return get_train_mask_greedy(
            self.label_percent, self.initial_train_mask, self.V_ksparse,
            self.V_ksparse_H, self.get_v, self.H, self.H_h, self.cov_x,
            self.cov_w, self.W, self.num_nodes)
