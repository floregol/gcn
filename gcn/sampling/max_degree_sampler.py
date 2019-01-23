from sampling.sampler import Sampler
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm
import numpy as np


class MaxDegreeSampler(Sampler):

    def __init__(self, initial_train_mask, adj):
        self.name = "MaxDegree"
        self.initial_train_mask = initial_train_mask
        self.adj = adj
        self.multi_trials = True
        super(MaxDegreeSampler, self).__init__(initial_train_mask, adj)

    def get_results_tuple(self, fileinfo, settings, result):
        super(MaxDegreeSampler, self).set_info(settings, result)
        results_filename = fileinfo + "maxdegree_results.p"
        return (self.dict_output, results_filename)

    def get_train_mask_fun(self, seed):
        np.random.seed(seed=seed)  # To garantee randomness between threads
        degree = np.sum(self.adj, axis=0)
        train_index = np.argwhere(self.initial_train_mask).reshape(-1)
        train_mask = np.zeros((initial_train_mask.shape), dtype=bool)  # list of False
       
        random_sampling_set_size = int(
            (label_percent / 100) * train_index.shape[0])
        random_list = random.sample(
            range(train_index.shape[0]), random_sampling_set_size)
        train_mask[random_list] = True
        label_percent = (100 * np.sum(train_mask) / train_index.shape[0])
        return train_mask, label_percent