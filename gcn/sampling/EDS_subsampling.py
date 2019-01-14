import numpy as np


def get_train_mask_EDS(label_percent, initial_train_mask, U_ksparse,
                       num_nodes):

    train_index = np.argwhere(initial_train_mask).reshape(-1)
    unscaled_probabilies = np.linalg.norm(U_ksparse[:, train_index], axis=0)
    probabilities = unscaled_probabilies / np.sum(unscaled_probabilies)

    #TODO filter by adj train
    train_mask = np.zeros(
        (initial_train_mask.shape), dtype=bool)  # list of False

    random_sampling_set_size = int(
        (label_percent / 100) * train_index.shape[0])
    EDS_subset = []
    
    remainin_nodes_to_sample = random_sampling_set_size
    while len(EDS_subset) < random_sampling_set_size:
        EDS_subset = EDS_subset + list(
            np.random.choice(
                train_index, remainin_nodes_to_sample, p=probabilities))
        EDS_subset = list(set(EDS_subset))
        remainin_nodes_to_sample = random_sampling_set_size - len(EDS_subset)

    train_mask[EDS_subset] = True
    mask = np.ones((initial_train_mask.shape), dtype=bool)
    mask[train_index] = 0
    train_mask[mask] = False
    label_percent = (100 * np.sum(train_mask) / train_index.shape[0])
   
    return train_mask, label_percent
