from sampling.sampler import Sampler
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random


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

    def precomputations(self):
        degree = np.sum(self.adj, axis=0)[0].tolist()[0]
        self.dict_degree_train_index = {}
        self.list_of_degree = []
        for i in self.train_index:  # Only consider the degree of the node in the taining set
            if degree[i] in self.dict_degree_train_index:
                self.dict_degree_train_index[degree[i]].append(
                    i)  # append to the list of index with this degree
            else:
                self.list_of_degree.append(degree[i])
                self.dict_degree_train_index[degree[i]] = [i]
        self.list_of_degree.sort(reverse=True)

    def get_train_mask_fun(self, seed):
        np.random.seed(seed=seed)  # To garantee randomness between threads
        train_mask = np.zeros(
            (self.initial_train_mask.shape), dtype=bool)  # list of False

        random_sampling_set_size = int(
            (self.label_percent / 100) * self.train_index.shape[0])
        max_degree_subset = []
        for degree in self.list_of_degree:
            list_node_with_degree = self.dict_degree_train_index[degree]
            if len(max_degree_subset) + len(
                    list_node_with_degree) <= random_sampling_set_size:
                max_degree_subset = max_degree_subset + list_node_with_degree
            else:  # Need to choose at random the nodes that will be added
                num_missing_nodes = random_sampling_set_size - len(
                    max_degree_subset)
                random_subset_node_with_degree = random.sample(
                    list_node_with_degree, num_missing_nodes)
                max_degree_subset = max_degree_subset + random_subset_node_with_degree
                break

        train_mask[max_degree_subset] = True
        mask = np.ones((self.initial_train_mask.shape), dtype=bool)
        mask[self.train_index] = 0
        train_mask[mask] = False
        label_percent = (100 * np.sum(train_mask) / self.train_index.shape[0])

        return train_mask, label_percent