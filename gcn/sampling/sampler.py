from sampling.greedy_subsampling import *
from sampling.EDS_subsampling import *
from train_gcnn_trials import train_results
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm


class Sampler:
    def __init__(self, initial_train_mask, adj):
        self.name = ""
        self.initial_train_mask = initial_train_mask
        self.adj = adj

    def get_train_mask_fun():
        pass


class EDS_Sampler(Sampler):
    def __init__(self, initial_train_mask, adj):
        self.name = "EDS"
        self.K_sparse = 0
        self.label_percent = 0
        self.U_ksparse = None
        self.num_nodes = 0
        super(EDS_Sampler, self).__init__(initial_train_mask, adj)

    def precomputations(self):
        self.noise = 10  # To delete
        _, self.U_ksparse, _, _, _, _, _, _, se   lf.num_nodes = greedy_eigenvector_precomputation(
            self.adj, self.K_sparse, self.noise, self.initial_train_mask)

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
