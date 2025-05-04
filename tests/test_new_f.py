import unittest

from algo.helpers import f_new, input_graph, pack_transit_node

import networkx as nx


class TestNewF(unittest.TestCase):

    def test_f_new_with_transit_node1(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1,), (1, 2), (2, 3)]

        for edge in edges:
            G.add_edge(*edge, weight=1, initial_edge=edge)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 0]

        f_val_1 = f_new(G, PG, partition)

        proc = 1
        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 1], 'Не то разбиение после запаковки')

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[2]['weight'], 2, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node2(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0]
        proc = 1

        f_val_1 = f_new(G, PG, partition)

        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 0, 0, 1], 'Не то разбиение после запаковки')

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[4]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node3(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7), (2, 6), (3, 6)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0]
        proc = 1

        f_val_1 = f_new(G, PG, partition)

        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 0, 0, 1], 'Не то разбиение после запаковки')

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[4]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        
    def test_f_new_with_transit_node4(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7), (2, 6), (3, 6), (8, 3), (9, 8)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
        proc = 1

        f_val_1 = f_new(G, PG, partition)

        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 0, 0, 0, 0, 1], 'Не то разбиение после запаковки')

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[6]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
