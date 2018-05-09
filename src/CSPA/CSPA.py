#!/usr/bin/env python

# Cluster_Ensembles/src/Cluster_Ensembles/Cluster_Ensembles.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Cluster_Ensembles is a package for combining multiple partitions 
into a consolidated clustering.
The combinatorial optimization problem of obtaining such a consensus clustering
is reformulated in terms of approximation algorithms for 
graph or hyper-graph partitioning.

References
----------
* Giecold, G., Marco, E., Trippa, L. and Yuan, G.-C.,
"Robust Lineage Reconstruction from High-Dimensional Single-Cell Data". 
ArXiv preprint [q-bio.QM, stat.AP, stat.CO, stat.ML]: http://arxiv.org/abs/1601.02748

* Strehl, A. and Ghosh, J., "Cluster Ensembles - A Knowledge Reuse Framework
for Combining Multiple Partitions".
In: Journal of Machine Learning Research, 3, pp. 583-617. 2002

* Kernighan, B. W. and Lin, S., "An Efficient Heuristic Procedure 
for Partitioning Graphs". 
In: The Bell System Technical Journal, 49, 2, pp. 291-307. 1970

* Karypis, G. and Kumar, V., "A Fast and High Quality Multilevel Scheme 
for Partitioning Irregular Graphs"
In: SIAM Journal on Scientific Computing, 20, 1, pp. 359-392. 1998

* Karypis, G., Aggarwal, R., Kumar, V. and Shekhar, S., "Multilevel Hypergraph Partitioning: 
Applications in the VLSI Domain".
In: IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 7, 1, pp. 69-79. 1999
"""
import metis
import gc
import numpy as np
import operator
import scipy.sparse
import warnings
import six
from six.moves import range
from functools import reduce

np.seterr(invalid = 'ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

def build_hypergraph_adjacency(cluster_runs):
    """Return the adjacency matrix to a hypergraph, in sparse matrix representation.
    
    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    hypergraph_adjacency : compressed sparse row matrix
        Represents the hypergraph associated with an ensemble of partitions,
        each partition corresponding to a row of the array 'cluster_runs'
        provided at input.
    """

    N_runs = cluster_runs.shape[0]

    hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
    for i in range(1, N_runs):
        hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
                                                   create_membership_matrix(cluster_runs[i])], 
                                                   format = 'csr')
    return hypergraph_adjacency

def calc_s(hypergraph_adjacency, num_runs):
    s = scipy.sparse.csr_matrix.dot(hypergraph_adjacency.transpose().tocsr(), hypergraph_adjacency)
    s = np.squeeze(np.asarray(s.todense()))
    
    del hypergraph_adjacency
    gc.collect()

    e_sum_before = s.sum()
    sum_after = 100000000.0  
    scale_factor = sum_after / float(e_sum_before)
    s = s * scale_factor
    s = np.rint(s[:])
    
    for i in range(s.shape[0]):
        s[i,i] = 0
    return s

def s_to_adjlist(s):
    from itertools import combinations
    num_nodes = s.shape[0]
    adjlist = [[] for _ in range(num_nodes)]
    for i, j in combinations(range(num_nodes), 2):
        if s[i][j] > 0:
            adjlist[i].append((j, int(s[i][j])))
            adjlist[j].append((i, int(s[i][j])))
    return adjlist


def CSPA(cluster_runs, N_clusters_max=None):
    hga = build_hypergraph_adjacency(cluster_runs)
    s = calc_s(hga, cluster_runs.shape[0])
    adjlist = s_to_adjlist(s)
    G = metis.adjlist_to_metis(adjlist)
    (edgecuts, parts) = metis.part_graph(G, N_clusters_max)
    return parts


def create_membership_matrix(cluster_run):
    """For a label vector represented by cluster_run, constructs the binary 
        membership indicator matrix. Such matrices, when concatenated, contribute 
        to the adjacency matrix for a hypergraph representation of an 
        ensemble of clusterings.
    
    Parameters
    ----------
    cluster_run : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    An adjacnecy matrix in compressed sparse row form.
    """

    cluster_run = np.asanyarray(cluster_run)

    if reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
        raise ValueError("\nERROR: Cluster_Ensembles: create_membership_matrix: "
                         "problem in dimensions of the cluster label vector "
                         "under consideration.")
    else:
        cluster_run = cluster_run.reshape(cluster_run.size)

        cluster_ids = np.unique(np.compress(np.isfinite(cluster_run), cluster_run))
      
        indices = np.empty(0, dtype = np.int32)
        indptr = np.zeros(1, dtype = np.int32)

        for elt in cluster_ids:
            indices = np.append(indices, np.where(cluster_run == elt)[0])
            indptr = np.append(indptr, indices.size)

        data = np.ones(indices.size, dtype = int)

        return scipy.sparse.csr_matrix((data, indices, indptr), shape = (cluster_ids.size, cluster_run.size))

