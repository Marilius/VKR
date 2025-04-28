import unittest

from algo.helpers import f_new, input_graph, pack_transit_node

import networkx as nx


class TestNewF(unittest.TestCase):

    def test_f_new_with_transit_node1(self):
        G = nx.MultiDiGraph()
        G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 0]

        f_val_1 = f_new(G, PG, partition)
        
        proc = 1
        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 1], 'Не то разбиение после запаковки')
        
        f_val_2 = f_new(G, PG, partition)
        
        self.assertEqual(G.nodes[2]['weight'], 2, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
