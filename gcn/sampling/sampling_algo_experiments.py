from sampling.greedy_subsampling import *
from sampling.EDS_subsampling import *
from train_gcnn_trials import train_results
from subsample import get_train_mask
from datetime import datetime
from tqdm import tqdm


# TODO generalized to every sampling experiment. Move stuf to sampler class
def sampling_experiment(trials, sampler, adj, initial_train_mask,
                        labels_percent_list, model_gcn, features, y_train,
                        y_val, y_test, val_mask, test_mask,
                        SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
                        settings, fileinfo, stats_adj_helper):
    K_sparse_list = [5, 10, 100]
    results_tuple = []
    for K_sparse in K_sparse_list:
        sampler.K_sparse = K_sparse
        sampler.precomputations()
        result = []
        for label_percent in labels_percent_list:
            sampler.label_percent = label_percent
            result.append(
                train_results(trials, sampler, label_percent, model_gcn, adj,
                              features, y_train, y_val, y_test,
                              initial_train_mask, val_mask, test_mask, True,
                              True, SHOW_TEST_VAL_DATASET_STATS,
                              VERBOSE_TRAINING, settings, stats_adj_helper))

        info = {
            'K_sparsity': K_sparse,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': settings
        }

        dict_output = {'results': result, 'info': info}
        results_filename = fileinfo + 'K_sparsity=' + str(
            K_sparse) + '_results.p'
        results_tuple.append((dict_output, results_filename))

    return results_tuple


# def greedy_sampling_experiments(trials, adj, initial_train_mask,
#                                 labels_percent_list, model_gcn, features,
#                                 y_train, y_val, y_test, val_mask, test_mask,
#                                 SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                                 settings, fileinfo, stats_adj_helper):
#     K_sparse_list = [5, 10, 100]
#     noise_list = [0.01, 1, 100]
#     results_tuple = []
#     for K_sparse in K_sparse_list:
#         for noise in noise_list:
#             result = []
#             V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, num_nodes = greedy_eigenvector_precomputation(
#                 adj, K_sparse, noise, initial_train_mask)

#             def get_train_mask_fun():
#                 return get_train_mask_greedy(label_percent, initial_train_mask,
#                                              V_ksparse, V_ksparse_H, get_v, H,
#                                              H_h, cov_x, cov_w, W, num_nodes)

#             for label_percent in labels_percent_list:
#                 result.append(
#                     train_results(
#                         1, label_percent, model_gcn, get_train_mask_fun, adj,
#                         features, y_train, y_val, y_test, initial_train_mask,
#                         val_mask, test_mask, True, True,
#                         SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                         settings, fileinfo, stats_adj_helper))

#             info = {
#                 'noise': noise,
#                 'K_sparsity': K_sparse,
#                 'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 'params': settings
#             }

#             dict_output = {'results': result, 'info': info}
#             results_filename = fileinfo + 'K_sparsity=' + str(
#                 K_sparse) + '_label_balance=Greedy_noise=' + str(
#                     noise) + '_results.p'
#             results_tuple.append((dict_output, results_filename))

#     return results_tuple

# def EDS_sampling_experiments(trials, adj, initial_train_mask,
#                              labels_percent_list, model_gcn, features, y_train,
#                              y_val, y_test, val_mask, test_mask,
#                              SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                              settings, fileinfo, stats_adj_helper):
#     K_sparse_list = [5, 10, 100]
#     results_tuple = []
#     noise = 100  # dummy value, not used
#     for K_sparse in K_sparse_list:
#         result = []
#         _, U_ksparse, _, _, _, _, _, _, num_nodes = greedy_eigenvector_precomputation(
#             adj, K_sparse, noise, initial_train_mask)

#         def get_train_mask_fun():
#             return get_train_mask_EDS(label_percent, initial_train_mask,
#                                       U_ksparse, num_nodes)

#         for label_percent in labels_percent_list:
#             result.append(
#                 train_results(trials, label_percent, model_gcn,
#                               get_train_mask_fun, adj, features, y_train,
#                               y_val, y_test, initial_train_mask, val_mask,
#                               test_mask, True, True,
#                               SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                               settings, fileinfo, stats_adj_helper))

#         info = {
#             'K_sparsity': K_sparse,
#             'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'params': settings
#         }

#         dict_output = {'results': result, 'info': info}
#         results_filename = fileinfo + 'K_sparsity=' + str(
#             K_sparse) + '_results.p'
#         results_tuple.append((dict_output, results_filename))

#     return results_tuple

# def random_sampling_experiments(trials, adj, initial_train_mask,
#                                 labels_percent_list, model_gcn, features,
#                                 y_train, y_val, y_test, val_mask, test_mask,
#                                 SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                                 settings, fileinfo, stats_adj_helper):
#     results_tuple = []
#     result = []
#     for label_percent in labels_percent_list:

#         def get_train_mask_fun():
#             return get_train_mask(label_percent, y_train, initial_train_mask,
#                                   True)

#         result.append(
#             train_results(trials, label_percent, model_gcn, get_train_mask_fun,
#                           adj, features, y_train, y_val, y_test,
#                           initial_train_mask, val_mask, test_mask, True, True,
#                           SHOW_TEST_VAL_DATASET_STATS, VERBOSE_TRAINING,
#                           settings, fileinfo, stats_adj_helper))

#         info = {
#             'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'params': settings
#         }

#         dict_output = {'results': result, 'info': info}
#         results_filename = fileinfo + "random_results.p"
#         results_tuple.append((dict_output, results_filename))

#     return results_tuple