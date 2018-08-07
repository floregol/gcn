import numpy as np
from itertools import combinations
from sampling_algo_util import *


def greedy_algo(V_ksparse: np.array, V_ksparse_H: np.array, get_v, H: np.array, H_h: np.array, cov_x: np.array,
                cov_w: np.array, W: np.array, number_node_sampled: int, num_nodes: int) -> list:
    # Variable initialisation
    G_subset = []
    remaining_node = list(range(0, num_nodes))
    K = cov_x

    for j in range(number_node_sampled):
        u = argmax(K, W, cov_w, remaining_node, get_v)
        K = update_K(K, W, cov_w, u, get_v)

        G_subset.append(u)  # Iterativly add a new node to the set
        remaining_node.remove(u)

    return G_subset


def brute_force_algo(V_ksparse: np.array, V_ksparse_H: np.array, get_v, H: np.array, H_h: np.array, cov_x: np.array,
                     cov_w: np.array, W: np.array, number_node_sampled: int, num_nodes: int) -> (list, dict):
    subset_scores = {}
    optimal_subset = []
    optimal_K_T = get_upper_bound_trace_K(W, cov_x)

    all_possible_set_combination = combinations(range(num_nodes), number_node_sampled)
    for possible_set in all_possible_set_combination:  # Try every subset
        score = get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, possible_set)
        subset_scores[str(list(possible_set))] = score
        if score <= optimal_K_T:
            optimal_K_T = score
            optimal_subset = possible_set

    # Add the empty set score to compute performance metric
    empty_set_score = get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, [])
    subset_scores[str([])] = empty_set_score

    return optimal_subset, subset_scores


def get_relative_suboptimality(optimal_K_T: float, f_K_T: float, empty_K_T: float) -> float:
    return (f_K_T - optimal_K_T) / (empty_K_T - optimal_K_T)