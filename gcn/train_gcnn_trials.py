from train import train_model
from subsample import *
from utils import *
from output_stats import *
from build_support import get_model_and_support


def train_results(trials, label_percent, model_gcn, get_train_mask_fun, adj, features, y_train, y_val, y_test,
                  initial_train_mask, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,
                  SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper):
    result = []
    acc_trials = np.zeros((trials,))
    for trial in range(trials):
        train_mask, label_percent = get_train_mask_fun()
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
    return (model_gcn, known_percent, acc_average, acc_var)
