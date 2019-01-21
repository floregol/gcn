from sampling.sampler import Sampler
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm
import numpy as np

class RandomSampler(Sampler):
    def __init__(self, initial_train_mask, adj, y_train):
        self.name = "Random"
        self.y_train = y_train
        super(RandomSampler, self).__init__(initial_train_mask, adj)

    def get_results_tuple(self, fileinfo, settings, result):
        super(RandomSampler, self).set_info(settings, result)
        results_filename = fileinfo + "random_results.p"
        return (self.dict_output, results_filename)

    def get_train_mask_fun(self, seed):
        np.random.seed(seed=seed)  # To garantee randomness between threads
        return get_train_mask(self.label_percent, self.y_train,
                              self.initial_train_mask, False)
