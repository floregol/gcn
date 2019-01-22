from train import train_model
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
    result = []
    acc_trials = np.zeros((trials, ))

    cores = 8
    num_iter = int(trials / cores)

    pool = mp.Pool(processes=cores)
    pool_results = [
        pool.apply_async(
            trial_run,
            (seed_indices, num_iter, label_percent, model_gcn, sampler, adj,
             features, y_train, y_val, y_test, initial_train_mask, val_mask,
             test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
             SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings,
             stats_adj_helper)) for seed_indices in range(cores)
    ]
    pool.close()
    pool.join()
    i_results = 0
    for pr in pool_results:
        thread_results = pr.get()
        acc_trials[i_results:i_results + len(thread_results)] = thread_results
        i_results = i_results + len(thread_results)

    print("----------------------------------")
    print(acc_trials)
    acc_average = np.mean(acc_trials)
    print(acc_average)
    acc_var = np.std(acc_trials)
    print(acc_var)
    print("----------------------------------")
    return (model_gcn, label_percent, acc_average, acc_var)


def trial_run(seed, num_iter, label_percent, model_gcn, sampler, adj, features,
              y_train, y_val, y_test, initial_train_mask, val_mask, test_mask,
              MAINTAIN_LABEL_BALANCE, WITH_TEST, SHOW_TEST_VAL_DATASET_STATS,
              VERBOSE_TRAINING, settings, stats_adj_helper):
    acc_iter = []
    for iter in range(num_iter):
        train_mask, label_percent = sampler.get_train_mask_fun(seed)
        print(train_mask)
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
    return acc_iter