from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import tensorflow as tf
import pickle as pk
import os
from subsample import *
from utils import *
from train import train_model
from output_stats import *
from build_support import get_model_and_support
from settings import set_tf_flags, graph_settings
from classification_stats import get_classification_stats
from graph_processing import *
"""
Class used to plot graph for multiple labels % . 
"""
result_folder = "results_greedy"


def train_and_save(trials, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val, y_test,
                   initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                   SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper):
    result = []
    acc_trials = np.zeros((trials,))
    for trial in range(trials):
        train_mask = get_train_mask_fun(label_percent, y_train, initial_train_mask, adj, MAINTAIN_LABEL_BALANCE)
        paths_to_known_list = get_num_paths_to_known(get_list_from_mask(train_mask), stats_adj_helper)
        print_partition_index(initial_train_mask, "Train", y_train)
        print_partition_index(val_mask, "Val", y_val)
        print_partition_index(test_mask, "Test", y_test)
        known_percent = show_data_stats(
            adj,
            features,
            y_train,
            y_val,
            y_test,
            train_mask,
            val_mask,
            test_mask,
        )
        model_func, support, sub_sampled_support, num_supports = get_model_and_support(
            model_gcn, adj, initial_train_mask, train_mask, val_mask, test_mask, WITH_TEST)

        test_acc, list_node_correctly_classified = train_model(
            model_func,
            num_supports,
            support,
            features,
            y_train,
            y_val,
            y_test,
            train_mask,
            val_mask,
            test_mask,
            sub_sampled_support,
            VERBOSE_TRAINING,
            seed=settings['seed'],
            list_adj=stats_adj_helper)
        acc_trials[trial] = test_acc
    print(acc_trials)
    acc_average = np.mean(acc_trials)
    print(acc_average)
    acc_var = np.std(acc_trials)
    print(acc_var)
    result.append((model_gcn, known_percent, acc_average, acc_var))
    correct_paths_to_known, incorrect_paths_to_known = get_classification_stats(list_node_correctly_classified,
                                                                                get_list_from_mask(test_mask),
                                                                                paths_to_known_list)
    info = {
        'MAINTAIN_LABEL_BALANCE': MAINTAIN_LABEL_BALANCE,
        'WITH_TEST': WITH_TEST,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'params': settings
    }

    dict_output = {
        'results': result,
        'info': info,
        'stats': {
            'correct_paths_to_known': correct_paths_to_known,
            'incorrect_paths_to_known': incorrect_paths_to_known
        }
    }
    pk.dump(dict_output,
            open(
                os.path.join(result_folder, fileinfo + 'w_test_features=' + str(WITH_TEST) + '_label_balance=' +
                             str(MAINTAIN_LABEL_BALANCE) + '_results.p'), 'wb'))


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
                           labels_percent_list,
                           SHOW_TEST_VAL_DATASET_STATS,
                           VERBOSE_TRAINING,
                           settings={},
                           fileinfo="",
                           stats_adj_helper=None):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    trials = 5
    for model_gcn in models_list:
        if model_gcn is 'gcn_greedy':
            print("gcn_greedy")
            for label_percent in labels_percent_list:
                train_and_save(1, label_percent, model_gcn, get_train_mask_greedy, adj, features, y_train, y_val,
                               y_test, initial_train_mask, val_mask, test_mask, True, True, SHOW_TEST_VAL_DATASET_STATS,
                               VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper)
        else:
            for MAINTAIN_LABEL_BALANCE in maintain_label_balance_list:
                for WITH_TEST in with_test_features_list:
                    for label_percent in labels_percent_list:
                        train_and_save(trials, label_percent, model_gcn, get_train_mask, adj, features, y_train, y_val,
                                       y_test, initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE,
                                       WITH_TEST, SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo,
                                       stats_adj_helper)


if __name__ == "__main__":
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    settings = graph_settings()['default']
    set_tf_flags(settings['params'], flags)
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

    labels_percent_list = [5, 10, 15, 20, 30, 50, 75]
    #list_adj = get_adj_powers(adj.toarray())
    list_adj = None
    maintain_label_balance_list = [True, False]
    with_test_features_list = [True]
    models_list = ['gcn_greedy', 'gcn']

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
        labels_percent_list,
        SHOW_TEST_VAL_DATASET_STATS,
        VERBOSE_TRAINING,
        settings=settings,
        stats_adj_helper=list_adj)
