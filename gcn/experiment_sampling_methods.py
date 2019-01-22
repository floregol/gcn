import tensorflow as tf
import pickle as pk
import os
from utils import *
from settings import set_tf_flags, graph_settings
from sampling.sampling_algo_experiments import sampling_experiment
from sampling.eds_sampler import EDSSampler
from sampling.random_sampler import RandomSampler
from sampling.greedy_sampler import GreedySampler
from sampling.max_degree_sampler import MaxDegreeSampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
Sampling experiment. 
Different sampling algorithms are tried to pick out nodes that will composed the labeled training set for GCN training.
The goal is to see if one technique is more sensible than an other.
"""
"""
This is the file to execute to run and save the experiments in the result folder.
"""
result_folder = "results_sampling"
"""
Settings                    : default -> for Kipf GCN settings, quick -> for running the whole thing fast (to check that everything works)
labels_percent_list         : determines how many labeled nodes will be used for training. Percent with respect to the training set, 100% means the whole training set
list adj                    : used to display stats on connection to known nodes of wrongly/correctly classified nodes.
maintain_label_balance_list : will (try) to keep the same ratio of classes in labeled training set
with_test_features_list     : Hides testing features by modifying the Adj matrix if set to False
models_list                 : Models to try. Subsample GCN strictly uses subsmaple nodes, GCN uses all the features

"""
if __name__ == "__main__":
    # Tensorflow settings
    flags = tf.app.flags

    FLAGS = flags.FLAGS
    settings = graph_settings()['default']
    set_tf_flags(settings['params'], flags, verbose=True)

    # Verbose settings
    SHOW_TEST_VAL_DATASET_STATS = False
    VERBOSE_TRAINING = False

    # Random seed
    seed = settings['seed']
    np.random.seed(seed)

    SAMPLING_TRIALS = 16  # How many time the same experiment will be repeated to get standard dev.
    # Load data. Features and labels.
    adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(
        FLAGS.dataset)
    # Some preprocessing
    features = preprocess_features(features)

    labels_percent_list = [5, 10, 15, 20, 30, 40, 50, 60, 75, 85, 100]
    #labels_percent_list = [30, 50]

    print(
        "Getting powers of the adjacency matrix A"
    )  # TODO optimize this step by using the already computer eigenvalues
    #list_adj = get_adj_powers(adj.toarray())
    list_adj = None
    stats_adj_helper = list_adj

    print("Finish getting powers of A")
    fileinfo = "Official"
    K_sparse_list = [5, 10, 100]
    noise_list = [0.01, 1, 100]
    maintain_label_balance_list = [False]
    with_test_features_list = [True]
    models_list = ['gcn']
    sampler_list = [
        EDSSampler(initial_train_mask, adj, K_sparse_list),
        GreedySampler(initial_train_mask, adj, K_sparse_list, noise_list),
        RandomSampler(initial_train_mask, adj, y_train),
        MaxDegreeSampler(initial_train_mask, adj)
    ]

    # Create result folders

    print("Saving results in folder " + result_folder)
    print()
    print("-------------------------------------")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Run the experiment
    for sampler in sampler_list:
        print()
        print("Sampling method : " + sampler.name)
        print("-------------------------------------")
        for model_gcn in models_list:
            # Run sampling experiment
            results_tuples = sampling_experiment(
                SAMPLING_TRIALS, sampler, adj, initial_train_mask,
                labels_percent_list, model_gcn, features, y_train, y_val,
                y_test, val_mask, test_mask, SHOW_TEST_VAL_DATASET_STATS,
                VERBOSE_TRAINING, settings, fileinfo, stats_adj_helper)
            # Save results
            for dict_output, results_filename in results_tuples:
                pk.dump(
                    dict_output,
                    open(
                        os.path.join(
                            result_folder,
                            settings['params']['dataset'] + "_sampling_method="
                            + sampler.name + "_" + results_filename), 'wb'))
