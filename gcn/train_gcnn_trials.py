from train import train_model
from subsample import *
from utils import *
from output_stats import *
from build_support import get_model_and_support
from tqdm import tqdm
import multiprocessing as mp


def train_results(trials, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val, y_test,
                  initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                  SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper):
    result = []
    acc_trials = np.zeros((trials,))

    cores = 4
    num_iter = int(trials / cores)

    pool = mp.Pool(processes=cores)
    pool_results = [
        pool.apply_async(trial_run,
                         (num_iter, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val, y_test,
                          initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                          SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper))
        for indices in range(cores)
    ]
    pool.close()
    pool.join()
    for pr in pool_results:
        dict_simul = pr.get()
        print(dict_simul['time'])
        result_dict['greedy'] += (dict_simul['greedy'])
        result_dict['deterministic'] += (dict_simul['deterministic'])
        result_dict['random_leverage'] += (dict_simul['random_leverage'])
        result_dict['uniform_random'] += (dict_simul['uniform_random'])

    print(acc_trials)
    acc_average = np.mean(acc_trials)
    print(acc_average)
    acc_var = np.std(acc_trials)
    print(acc_var)
    return (model_gcn, label_percent, acc_average, acc_var)


def trial_run(num_iter, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val, y_test,
              initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST, SHOW_TEST_VAL_DATASET_STATS,
              VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper):
    for trial in range(trials):
        train_mask, label_percent = get_train_mask_fun()
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