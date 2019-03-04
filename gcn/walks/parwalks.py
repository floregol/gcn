import numpy as np
from numpy.linalg import inv
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg


class ParWalks:

    def __init__(self, adj):
        self.adj = adj.todense()
        self.alpha = 10e-6

    def get_probabilities(self, train_mask, labels, initial_train_mask, stored_A=None):
        y_train = np.zeros(labels.shape)
        train_index = np.where(train_mask)[0]
        y_train[train_index] = labels[train_index, :]
        A = self.absorption_probability(self.adj, self.alpha, stored_A, train_mask)
        already_labeled = np.sum(y_train, axis=1)
        P = np.zeros(labels.shape, dtype=np.float32)

        for i in range(y_train.shape[1]):
            y = y_train[:, i:i + 1]
            a = A.dot(y)
            a[already_labeled > 0] = 0
            P[:, i] = a[:, 0]
        return P

    def absorption_probability(self, W, alpha, stored_A=None, column=None):
        n = W.shape[0]
        print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)

        if column is not None:
            A = np.zeros(W.shape)
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            return A
        else:
            A = slinalg.inv(L).toarray()
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A
