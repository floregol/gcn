import pickle as pk
import os
from utils import *
from sampling.eigen_utils import get_P_matrix

"""
Reconstruction experiment. 

"""
"""
This is the file to execute to run and save the experiments in the result folder.
"""
result_folder = "reconstruction"

if __name__ == "__main__":

    # Random seed
    seed = 56
    np.random.seed(seed)

    SAMPLING_TRIALS = 20  # How many time the same experiment will be repeated to get standard dev.
    # Load data. Features and labels.
    adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(
        "cora")
    # Some preprocessing
    features = preprocess_features(features)

    labels_percent_list = [5, 10, 15, 20, 30, 40, 50, 60, 75, 85, 100]
    #labels_percent_list = [15,30, 50, 100]

    print("Computing absortion matrix P")
    P = get_P_matrix(adj, 0.00006)
   
    Create result folders

    print("Saving results in folder " + result_folder)
    print()
    print("-------------------------------------")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Run the experiment
    for label_generator in label_generator_list:
        print()
        print("Label Reconstruction method : " + sampler.name)
        print("-------------------------------------")
        for model_gcn in models_list:
            # Run sampling experiment
            results_tuples = sampling_experiment(
                SAMPLING_TRIALS, sampler, adj, initial_train_mask,
                labels_percent_list, model_gcn, features, y_train, y_val,
                y_test, val_mask, test_mask, MAINTAIN_LABEL_BALANCE, WITH_TEST,SHOW_TEST_VAL_DATASET_STATS,
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
