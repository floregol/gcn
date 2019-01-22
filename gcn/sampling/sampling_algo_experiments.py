
from train_gcnn_trials import train_results
from tqdm import tqdm

# TODO generalized to every sampling experiment. Move stuf to sampler class
def sampling_experiment(trials, sampler, adj, initial_train_mask,
                        labels_percent_list, model_gcn, features, y_train,
                        y_val, y_test, val_mask, test_mask,
                        SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
                        settings, fileinfo, stats_adj_helper):

    results_tuple = []
    while (sampler.next_parameter()):
        sampler.precomputations()
        result = []
        for label_percent in tqdm(labels_percent_list):
            
            sampler.label_percent = label_percent
            result.append(
                train_results(trials, sampler, label_percent, model_gcn, adj,
                              features, y_train, y_val, y_test,
                              initial_train_mask, val_mask, test_mask, True,
                              True, SHOW_TEST_VAL_DATASET_STATS,
                              VERBOSE_TRAINING, settings, stats_adj_helper))

        results_tuple.append(
            sampler.get_results_tuple(fileinfo, settings, result))
  
    return results_tuple

