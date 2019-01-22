from sampling.sampler import Sampler
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm
import numpy as np


class MaxDegreeSampler(Sampler):
    def __init__(self, initial_train_mask, adj):
        self.name = "MaxDegree"
        self.multi_trials = True
        super(MaxDegreeSampler, self).__init__(initial_train_mask, adj)

    def get_results_tuple(self, fileinfo, settings, result):
        super(MaxDegreeSampler, self).set_info(settings, result)
        results_filename = fileinfo + "maxdegree_results.p"
        return (self.dict_output, results_filename)

    def get_train_mask_fun(self, seed):
        np.random.seed(seed=seed)  # To garantee randomness between threads
        pass
