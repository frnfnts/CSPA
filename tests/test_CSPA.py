import unittest
import numpy as np
import metis
from src import CSPA

class CspaTest(unittest.TestCase):
    def test_create_membership_matrix(self):
        a = [1, 1, 2, 3, 2, 2, 3]
        res  = CSPA.create_membership_matrix(a).toarray()
        self.assertEqual(list(res[0]), [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(list(res[1]), [0, 0, 1, 0, 1, 1, 0])
        self.assertEqual(list(res[2]), [0, 0, 0, 1, 0, 0, 1])

    def test_build_hypergraph_adjacency(self):
        cluster_runs = np.array([[1,1,1,2,2,2], [1,1,2,2,2,2]])
        hga = CSPA.build_hypergraph_adjacency(cluster_runs)

    def test_calc_s(self):
        cluster_runs = np.array([[1,1,1,2,2,2], [1,1,2,2,2,2]])
        hga = CSPA.build_hypergraph_adjacency(cluster_runs)
        s = CSPA.calc_s(hga, cluster_runs.shape[0])
        self.assertEqual(s.shape[0], s.shape[1])
        self.assertEqual(s.shape[0], cluster_runs[0].shape[0])

    def test_s_to_adjlist(self):
        cluster_runs = np.array([[1,1,1,2,2,2], [1,1,2,2,2,2]])
        hga = CSPA.build_hypergraph_adjacency(cluster_runs)
        s = CSPA.calc_s(hga, cluster_runs.shape[0])
        adjlist = CSPA.s_to_adjlist(s)
        self.assertEqual(len(adjlist[0]), 2)
        self.assertEqual(len(adjlist[1]), 2)
        self.assertEqual(len(adjlist[2]), 5)
        self.assertEqual(len(adjlist[3]), 3)
        self.assertEqual(len(adjlist[4]), 3)
        self.assertEqual(len(adjlist[5]), 3)


    def test_adjlist_to_metis(self):
        cluster_runs = np.array([[1,1,1,2,2,2], [1,1,2,2,2,2]])
        hga = CSPA.build_hypergraph_adjacency(cluster_runs)
        s = CSPA.calc_s(hga, cluster_runs.shape[0])
        adjlist = CSPA.s_to_adjlist(s)
        G = metis.adjlist_to_metis(adjlist)

        self.assertEqual(G.nvtxs.value, 6)
        self.assertEqual(len(G.adjncy), 18)

    def test_CSPA(self):
        cluster_runs = np.array([[1,1,1,2,2,2], [1,1,2,2,2,2]])
        parts = CSPA.CSPA(cluster_runs, 2)
        self.assertEqual(parts, [0,0,0, 1,1,1])
