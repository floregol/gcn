import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from scipy.stats import entropy
import math


def score_vs_uncertainty(type_score, uncertainty_val, bins, predicted_labels, y_train):
    sorted_uncertainty_index = np.argsort(uncertainty_val)
    sorted_uncertainty = [uncertainty_val[i] for i in sorted_uncertainty_index]
    digitized = np.digitize(sorted_uncertainty, bins)
    x = list(bins)
    bin_dict = {}
    real_index = []
    for i in range(1, len(bins)):
        bin_index = np.where(digitized == i)
        real_index = sorted_uncertainty_index[bin_index]
        if np.sum(bin_index) > 0:
            score = accuracy_score(y_train[real_index], predicted_labels[real_index])
            bin_dict[x[i]] = score

    return bin_dict


def normalize_uncertainty_list(l, ignore_max=None):
    max_after = max(l)
    if ignore_max:
        max_after = max(list(filter(lambda x: x < ignore_max, l)))
        l = [i if i < ignore_max else max_after for i in l]

    normalized_list = (l - min(l)) / (max_after - min(l))
    return normalized_list


def reconstructed_labels_scores(y_train, P, num_new_nodes_per_class):
    y_reconstructed_true = np.zeros((y_train.shape[1] * num_new_nodes_per_class, y_train.shape[1]))
    y_reconstructed_pred = np.zeros((y_train.shape[1] * num_new_nodes_per_class, y_train.shape[1]))
    for i in range(y_train.shape[1]):
        a = P[:, i]
        index = np.argsort(-a)[0:num_new_nodes_per_class]
        y_index = num_new_nodes_per_class * i
        y_reconstructed_pred[y_index:y_index + num_new_nodes_per_class, i] = 1
        y_reconstructed_true[y_index:y_index + num_new_nodes_per_class, :] = y_train[index]

    f1_macro = f1_score(y_reconstructed_true, y_reconstructed_pred, average='macro')
    f1_micro = f1_score(y_reconstructed_true, y_reconstructed_pred, average='micro')
    acc = accuracy_score(y_reconstructed_true, y_reconstructed_pred)
    return f1_macro, f1_micro, acc


def compute_label_prop(y_train, train_mask, initial_train_mask, P, num_new_nodes_per_class, bins):

    f1_macro, f1_micro, acc = reconstructed_labels_scores(y_train[initial_train_mask, :], P[initial_train_mask, :],
                                                          num_new_nodes_per_class)
    labeled_index = np.argwhere(train_mask)
    filtered_train_mask = train_mask[np.argwhere(initial_train_mask).flatten()]
    unlabled_index = np.where(filtered_train_mask == False)[0]

    # Compute entroy/ magnitude at every node
    P = normalize(P,norm='l1', axis =0)
   
    probabilities = normalize(P, norm='l1', axis=1)

    max_entropy = int(math.log(P.shape[1]) * 100)

    def to_entropy(p_vector):
        if np.sum(p_vector) == 0:  # max entropy
            return max_entropy
        score = entropy(p_vector)
        if score == 0:
            return max_entropy
        return score

    def to_magnitude(p_vector):
        if np.sum(p_vector) == 0:  # max uncertainty
            return 0
        return np.sum(p_vector)

    entropy_list = np.apply_along_axis(to_entropy, 1, probabilities)
    magnitude_list = np.apply_along_axis(to_magnitude, 1, P)

    # Remove known nodes
    filtered_entropy_list = entropy_list[unlabled_index]
    # Remove nodes with 0 uncertainty (artificial)
    filtered_magnitude_list = magnitude_list[unlabled_index]

    normalized_entropy_list = normalize_uncertainty_list(filtered_entropy_list, max_entropy)
    normalized_magnitude = normalize_uncertainty_list(filtered_magnitude_list)
    normalized_magnitude_inverse = [1 - magnitude for magnitude in normalized_magnitude]

    predicted_labels = np.argmax(P, axis=1)
    predicted_labels_one_hot = np.zeros(y_train.shape)
    predicted_labels_one_hot[np.arange(y_train.shape[0]), predicted_labels] = 1

    # vector with correspond label vectors
    filtered_y_train = y_train[unlabled_index, :]
    filtered_predicted_labels_one_hot = predicted_labels_one_hot[unlabled_index, :]

    score_entro = score_vs_uncertainty(f1_macro, normalized_entropy_list, bins, filtered_predicted_labels_one_hot,
                                       filtered_y_train)

    score_magnitude = score_vs_uncertainty(f1_macro, normalized_magnitude_inverse, bins,
                                           filtered_predicted_labels_one_hot, filtered_y_train)

    # print("F1-score macro new nodes : " + str(f1_macro))
    # print("F1-score micro new nodes : " + str(f1_micro))
    # print("F1-score accuracy new nodes : " + str(acc))

    return (f1_macro, f1_micro, acc, score_entro, score_magnitude)
