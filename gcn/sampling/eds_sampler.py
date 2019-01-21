from sampling.greedy_subsampling import *
from sampling.sampler import Sampler
from datetime import datetime
from tqdm import tqdm


class EDSSampler(Sampler):
    def __init__(self, initial_train_mask, adj, K_sparse_list):
        self.name = "EDS"
        self.K_sparse_list = K_sparse_list
        self.K_sparse = 0
        self.U_ksparse = None
        super(EDSSampler, self).__init__(initial_train_mask, adj)

    def precomputations(self):
        self.noise = 10  # To delete
        _, self.U_ksparse, _, _, _, _, _, _, self.num_nodes = greedy_eigenvector_precomputation(
            self.adj, self.K_sparse, self.noise, self.initial_train_mask)

    def next_parameter(self):
        if (self.sampling_config_index == len(self.K_sparse_list)):
            return False
        self.K_sparse = self.K_sparse_list[self.sampling_config_index]
        self.sampling_config_index += 1
        return True

    def get_results_tuple(self, fileinfo, settings, result):
        super(EDSSampler, self).set_info(settings, result)
        self.info['K_sparsity'] = self.K_sparse
        results_filename = fileinfo + 'K_sparsity=' + str(
            self.K_sparse) + '_results.p'
        return (self.dict_output, results_filename)

    def get_train_mask_fun(self, seed):
        np.random.seed(seed=seed)  # To garantee randomness between threads
        train_index = np.argwhere(self.initial_train_mask).reshape(-1)
        unscaled_probabilies = np.linalg.norm(
            self.U_ksparse[:, train_index], axis=0)
        probabilities = unscaled_probabilies / np.sum(unscaled_probabilies)

        #TODO filter by adj train ?
        train_mask = np.zeros(
            (self.initial_train_mask.shape), dtype=bool)  # list of False

        random_sampling_set_size = int(
            (self.label_percent / 100) * train_index.shape[0])
        EDS_subset = []

        remainin_nodes_to_sample = random_sampling_set_size
        while len(EDS_subset) < random_sampling_set_size:
            EDS_subset = EDS_subset + list(
                np.random.choice(
                    train_index, remainin_nodes_to_sample, p=probabilities))
            EDS_subset = list(set(EDS_subset))
            remainin_nodes_to_sample = random_sampling_set_size - len(
                EDS_subset)

        train_mask[EDS_subset] = True
        mask = np.ones((self.initial_train_mask.shape), dtype=bool)
        mask[train_index] = 0
        train_mask[mask] = False
        label_percent = (100 * np.sum(train_mask) / train_index.shape[0])

        return train_mask, label_percent
