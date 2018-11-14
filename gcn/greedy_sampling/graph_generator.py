import numpy as np
import networkx as nx
from itertools import combinations
from random import randrange, uniform
from numpy.linalg import inv
import scipy.sparse as sp


def get_sparse_eigen_decomposition(graph, K):
    adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()), weight='weight').toarray()
    return get_sparse_eigen_decomposition_from_adj(adj, K)


def get_sparse_eigen_decomposition_from_adj(adj, K):
    eigenval, eigenvectors = np.linalg.eig(adj)
    eigenval_Ksparse = np.argsort(eigenval)[-K:]  # Find top eigenvalues index (not absolute values)
    V_ksparse = np.zeros((adj.shape[0], K))  # Only keep the eigenvectors of the max eigenvalues
    V_ksparse[:, 0:K] = eigenvectors[:, eigenval_Ksparse]
    V_ksparse = np.matrix(V_ksparse)
    V_ksparse_H = V_ksparse.getH()

    def get_v(index):
        v_index = V_ksparse_H[:, index]
        v_index_H = V_ksparse[index, :]
        return v_index, v_index_H

    return V_ksparse, V_ksparse_H, get_v


# Plotting graphs
def plot_graph(graph):
    nx.draw_shell(
        graph,
        with_labels=True,
    )


# Erdos Renyi graph: Add an edge with prob = 0.2
def generate_Erdos_Renyi_graph(n):
    Erdos_Renyi_Prob = 0.2
    Erdos_Renyi_graph = nx.erdos_renyi_graph(n, Erdos_Renyi_Prob)
    return Erdos_Renyi_graph


# Preferential attachment model
def generate_pref_attachment_graph(n):
    m0 = 1
    Pref_Attach_graph = nx.barabasi_albert_graph(n, m0)
    return Pref_Attach_graph


# Random graph with weight between [0,1]
def generate_random_graph(n):
    Random_graph = nx.Graph()
    for node_pair in combinations(range(n), 2):  # Generate each possible egde
        weight = uniform(0, 1.000001)  # Need to be [0,1]
        if weight > 0:
            Random_graph.add_edge(node_pair[0], node_pair[1], weight=weight)
    if (len(Random_graph.nodes()) < n):  # Recursivly generate a new graph until all nodes are connected
        del Random_graph  #  Delete the graph to avoid using memeroy unnecessarily
        return generate_random_graph(n)
    return Random_graph