from utils import *
import numpy as np
import pickle as pk
from walks.parwalks import ParWalks
from sklearn.model_selection import StratifiedShuffleSplit
from sampling.random_sampler import RandomSampler
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from walks.label_propagation import compute_label_prop

dataset = "cora"
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
result_folder = "pickle/walks_results"
seed = 32
np.random.seed(seed)
TRIALS = 10
n = features.shape[0]
known_labels = [2, 5, 10, 20, 40, 50, 85]
num_new_nodes_per_class = 10
resolution = 21
MAINTAIN_LABEL_BALANCE = True
NUM_CROSS_VAL = 4
TEST_SIZE = 0.369
TRAIN_SIZE = 0.7073
VERBOSE_TRAINING = True

ParWalks = ParWalks(adj)

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=TEST_SIZE, random_state=seed)
test_split.get_n_splits(labels, labels)

i = 1
for train_index, test_index in test_split.split(labels, labels):
    splitted_train_index = int(len(train_index) * TRAIN_SIZE)

    initial_train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    initial_train_mask[train_index[0:splitted_train_index]] = True
    val_mask[train_index[splitted_train_index:]] = True
    test_mask[test_index] = True

    y_train = np.zeros(labels.shape, dtype=int)
    y_val = np.zeros(labels.shape, dtype=int)
    y_test = np.zeros(labels.shape, dtype=int)
    y_train[initial_train_mask, :] = labels[initial_train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    seed_list = 100 * np.random.random_sample(TRIALS)

    sampler_list = [RandomSampler(initial_train_mask, adj, y_train, MAINTAIN_LABEL_BALANCE)]
    for sampler in sampler_list:
        print()
        print("Sampling method : " + sampler.name)
        print("-------------------------------------")
        for label_percent in known_labels:
            print()
            print("Label percent : " + str(label_percent))
            print("----------------------")
            sampler.label_percent = label_percent

            f1_macro_list = []
            f1_micro_list = []
            acc_list = []
            score_entro_list = []
            score_magnitude_list = []
            for trial in range(TRIALS):
                print()
                print("Trial : " + str(trial) + " seed : " + str(int(seed_list[trial])))

                train_mask, real_label_percent = sampler.get_train_mask_fun(int(seed_list[trial]))
                real_label_percent = "{:.1f}".format(real_label_percent)
                print(real_label_percent)
                P = ParWalks.get_probabilities(train_mask, labels, initial_train_mask)
                bins = np.linspace(0, 1, resolution)
                results_trial = compute_label_prop(y_train, train_mask, initial_train_mask, P, num_new_nodes_per_class,
                                                   bins)

                f1_macro_list.append(results_trial[0])
                f1_micro_list.append(results_trial[1])
                acc_list.append(results_trial[2])
                score_entro_list.append(results_trial[3])
                score_magnitude_list.append(results_trial[4])

            avg_f1_macro = np.mean(f1_macro_list)
            std_f1_macro = np.std(f1_macro_list)

            avg_f1_micro = np.mean(f1_micro_list)
            std_f1_micro = np.std(f1_micro_list)

            avg_acc = np.mean(acc_list)
            std_avg_acc = np.std(acc_list)

            avg_normalized_entropy_list = squash_list(bins, score_entro_list)
            avg_magnitude_list = squash_list(bins, score_magnitude_list)

            # avg_normalized_entropy_list =
            # avg_magnitude_list =
            scores = {
                "f1_macro": avg_f1_macro,
                "f1_micro": avg_f1_micro,
                "acc": avg_acc,
                "std_f1_macro": std_f1_macro,
                "std_f1_micro": std_f1_micro,
                "std_acc": std_avg_acc
            }
            results_dict = {
                "normalized_entropy_list": avg_normalized_entropy_list,
                "magnitude_list": avg_magnitude_list,
                "scores": scores,
                "label_percent" : real_label_percent
            }
            pk.dump(results_dict,
                    open(
                        os.path.join(result_folder, "partition_"+str(i)+"_sampler_" + sampler.name + "_label_" + str(label_percent) +"_added"+str(num_new_nodes_per_class)+ ".pk"),
                        'wb'))

    i += 1
