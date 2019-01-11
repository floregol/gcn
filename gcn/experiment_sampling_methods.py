from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import tensorflow as tf
import pickle as pk
import os
from train_gcnn_trials import train_results
from utils import *

from settings import set_tf_flags, graph_settings
from classification_stats import get_classification_stats
from graph_processing import *
from greedy_subsampling import *
import multiprocessing as mp
"""
Class used to plot graph for multiple labels % . 
"""
result_folder = "results_greedy"


def train_and_save_results(adj,
                           features,
                           y_train,
                           y_val,
                           y_test,
                           initial_train_mask,
                           val_mask,
                           test_mask,
                           maintain_label_balance_list,
                           with_test_features_list,
                           models_list,
                           sampling_list,
                           labels_percent_list,
                           SHOW_TEST_VAL_DATASET_STATS,
                           VERBOSE_TRAINING,
                           settings={},
                           fileinfo="",
                           stats_adj_helper=None):
    # Create result folders
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    trials = 5
    K_sparse_list = [5, 10, 100]
    noise_list = [0.01, 1, 100]
    for sampling_method in sampling_list:
        for model_gcn in models_list:
            for K_sparse in K_sparse_list:
                for noise in noise_list:
                    result = []
                    V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, num_nodes = greedy_eigenvector_precomputation(
                        adj, K_sparse, noise, initial_train_mask)

                    def get_train_mask_fun():
                        return get_train_mask_greedy(label_percent, initial_train_mask, V_ksparse, V_ksparse_H, get_v, H,
                                                    H_h, cov_x, cov_w, W, num_nodes)

                    for label_percent in labels_percent_list:
                        result.append(
                            train_results(1, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val,
                                        y_test, initial_train_mask, val_mask, test_mask, True, True,
                                        SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo,
                                        stats_adj_helper))

                    info = {
                        'noise': noise,
                        'K_sparsity': K_sparse,
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'params': settings
                    }

                    dict_output = {'results': result, 'info': info}
                    pk.dump(dict_output,
                            open(
                                os.path.join(result_folder, fileinfo + 'K_sparsity=' + str(K_sparse) +
                                            '_label_balance=Greedy_noise=' + str(noise) + '_results.p'), 'wb'))
                    # else:
                    #     for MAINTAIN_LABEL_BALANCE in maintain_label_balance_list:
                    #         for WITH_TEST in with_test_features_list:
                    #             for label_percent in labels_percent_list:

                    #                 def get_train_mask_fun():
                    #                     return get_train_mask(label_percent, y_train, initial_train_mask, MAINTAIN_LABEL_BALANCE)

                    #                 train_and_save(trials, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train,
                    #                                y_val, y_test, initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE,
                    #                                WITH_TEST, SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo,
                    #                                stats_adj_helper)


"""
Settings                    : default for Kipf settings, quick for testing purposes
labels_percent_list         : determines how many nodes will be used for training
list adj                    : used to display stats on connection to known nodes of wrongly/correctly classified nodes.
maintain_label_balance_list : will (try) to keep the same ratio of classes in training set
with_test_features_list     : Hides testing features by modifying the Adj matrix if set to False
models_list                 : Models to try. Subsample GCN strictly uses subsmaple nodes, GCN uses all the features

"""
if __name__ == "__main__":
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    settings = graph_settings()['quick']
    set_tf_flags(settings['params'], flags, verbose=True)

    # Verbose settings
    SHOW_TEST_VAL_DATASET_STATS = False
    VERBOSE_TRAINING = False

    # Random seed
    seed = settings['seed']
    np.random.seed(seed)

    # Load data
    adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    # Some preprocessing
    features = preprocess_features(features)

    labels_percent_list = [5, 10, 15, 20, 30, 40, 50, 60, 75, 85, 100]
    #labels_percent_list = [30,50]

    #list_adj = get_adj_powers(adj.toarray())
    list_adj = None

    maintain_label_balance_list = [True, False]
    with_test_features_list = [True]
    models_list = ['gcn']
    sampling_list = ['random', 'greedy', 'EDS', 'degree']

    # RUN
    train_and_save_results(
        adj,
        features,
        y_train,
        y_val,
        y_test,
        initial_train_mask,
        val_mask,
        test_mask,
        maintain_label_balance_list,
        with_test_features_list,
        models_list,
        sampling_list,
        labels_percent_list,
        SHOW_TEST_VAL_DATASET_STATS,
        VERBOSE_TRAINING,
        settings=settings,
        stats_adj_helper=list_adj)
