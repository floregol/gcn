from train import train_model
import numpy as np
from subsample import *
from utils import *
from output_stats import *
from build_support import get_model_and_support
from tqdm import tqdm
import multiprocessing as mp


def train_results(trials, sampler, label_percent, model_gcn, adj, features,
                  y_train, y_val, y_test, initial_train_mask, val_mask,
                  test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                  SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings,
                  stats_adj_helper):
    if sampler.multi_trials:
        result = []
        acc_trials = np.zeros((trials, ))
        actual_label_percent_trials = np.zeros((trials, ))
        cores = 8
        num_iter = int(trials / cores)

        pool = mp.Pool(processes=cores)
        pool_results = [
            pool.apply_async(
                trial_run,
                (seed_indices, num_iter, label_percent, model_gcn, sampler,
                 adj, features, y_train, y_val, y_test, initial_train_mask,
                 val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                 SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings,
                 stats_adj_helper)) for seed_indices in range(cores)
        ]
        pool.close()
        pool.join()
        i_results = 0
        for pr in pool_results:
            thread_results = pr.get()
            actual_label_percent_trials[
                i_results:i_results + len(thread_results)] = thread_results[1]
            acc_trials[i_results:
                       i_results + len(thread_results)] = thread_results[0]
            i_results = i_results + len(thread_results)

        acc_average = np.mean(acc_trials)
        acc_var = np.std(acc_trials)
        label_average = np.mean(actual_label_percent_trials)
        label_var = np.std(actual_label_percent_trials)

        print("----")
        print("{:.1f}".format(label_average), "% +/-",
              "{:.2f}".format(label_var), " Average accuracy :",
              "{:.3f}".format(acc_average), " +/- ", "{:.2f}".format(acc_var))
        print("----")
    else:
        num_iter = 1
        seed = 0
        acc_iter, label_iter = trial_run(
            seed, num_iter, label_percent, model_gcn, sampler, adj, features,
            y_train, y_val, y_test, initial_train_mask, val_mask, test_mask,
            MAINTAIN_LABEL_BALANCE, WITH_TEST, SHOW_TEST_VAL_DATASET_STATS,
            VERBOSE_TRAINING, settings, stats_adj_helper)
        acc_average = acc_iter[0]
        label_average = label_iter[0]
        acc_var = 0
        print("----")
        print("{:.1f}".format(label_average), "% Average accuracy :",
              "{:.3f}".format(acc_average))
        print("----")
    return (model_gcn, label_average, acc_average, acc_var)


def trial_run(seed, num_iter, label_percent, model_gcn, sampler, adj, features,
              y_train, y_val, y_test, initial_train_mask, val_mask, test_mask,
              MAINTAIN_LABEL_BALANCE, WITH_TEST, SHOW_TEST_VAL_DATASET_STATS,
              VERBOSE_TRAINING, settings, stats_adj_helper):
    acc_iter = []
    label_iter = []
    for iter in range(num_iter):
        train_mask, label_percent = sampler.get_train_mask_fun(seed)
        if VERBOSE_TRAINING:
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
            model_gcn, adj, initial_train_mask, train_mask, val_mask,
            test_mask, WITH_TEST)

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
        acc_iter.append(test_acc)
        label_iter.append(label_percent)
    return acc_iter, label_iter