import unittest

from algo.helpers import f_new, input_graph

import networkx as nx


class TestNewFBasic(unittest.TestCase):

    def test1(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (0, 2), (2, 1), (1, 3), (2, 3)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 0, 0, 0]

        f_val = f_new(G, PG, partition)
        self.assertEqual(f_val, 1.0, 'Значение ЦФ не то')

    def test2(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (0, 2), (2, 1), (1, 3), (2, 3)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 0, 1]

        f_val = f_new(G, PG, partition)
        self.assertEqual(f_val, 3.5, 'Значение ЦФ не то')

    def test3(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (1, 2), (2, 3)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 0, 1]

        f_val = f_new(G, PG, partition)
        self.assertEqual(f_val, 5.5, 'Значение ЦФ не то')
