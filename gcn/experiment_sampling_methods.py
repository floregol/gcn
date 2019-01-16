import tensorflow as tf
import pickle as pk
import os
from utils import *
from settings import set_tf_flags, graph_settings
from sampling.sampling_algo_experiments import sampling_experiment
from sampling.sampler import EDS_Sampler
"""
Class used to plot graph for multiple labels % . 
"""
result_folder = "results_sampling"
"""
Settings                    : default -> for Kipf GCN settings, quick -> for running the whole thing fast (to check that everything works)
labels_percent_list         : determines how many nodes will be used for training. Percent with respect to the training set, 100% means the whole training set
list adj                    : used to display stats on connection to known nodes of wrongly/correctly classified nodes.
maintain_label_balance_list : will (try) to keep the same ratio of classes in training set
with_test_features_list     : Hides testing features by modifying the Adj matrix if set to False
models_list                 : Models to try. Subsample GCN strictly uses subsmaple nodes, GCN uses all the features

"""
if __name__ == "__main__":
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    settings = graph_settings()['default']
    set_tf_flags(settings['params'], flags, verbose=True)

    # Verbose settings
    SHOW_TEST_VAL_DATASET_STATS = False
    VERBOSE_TRAINING = False
    print(settings)
    # Random seed
    seed = settings['seed']
    np.random.seed(seed)

    TRIALS = 20
    # Load data
    adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(
        FLAGS.dataset)
    # Some preprocessing
    features = preprocess_features(features)

    labels_percent_list = [5, 10, 15, 20, 30, 40, 50, 60, 75, 85, 100]
    #labels_percent_list = [30, 50]
    print("Getting A^p")
    #list_adj = get_adj_powers(adj.toarray())
    list_adj = None
    maintain_label_balance_list = [False]
    with_test_features_list = [True]
    models_list = ['gcn']
    sampler_list = [
        EDS_Sampler(initial_train_mask)
        # , ('random', random_sampling_experiments),
        #                  ('greedy', greedy_sampling_experiments), ('degree', None)
    ]

   
    # Create result folders
    print("Saving results in folder " + result_folder)
    print()
    print("-------------------------------------")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Run the experiment 
    for sampler in sampler_list:
        print("Sampling method : " + sampler.name)
        print("-------------------------------------")
        for model_gcn in models_list:
            # Run sampling experiment
            results_tuples = sampling_experiment(
                trials, sampler, adj, initial_train_mask, labels_percent_list,
                model_gcn, features, y_train, y_val, y_test, val_mask,
                test_mask, SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
                settings, fileinfo, stats_adj_helper)
            # Save results 
            for dict_output, results_filename in results_tuples:
                pk.dump(
                    dict_output,
                    open(
                        os.path.join(
                            result_folder,
                            settings['params']['dataset'] + "_sampling_method="
                            + sampling_method + "_" + results_filename), 'wb'))
