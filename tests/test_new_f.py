import unittest

from algo.helpers import f_new, input_graph, pack_transit_node, findLongestPath

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

    def test_f_new_with_transit_node5(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
        proc = 1

        f_val_1 = f_new(G, PG, partition)

        pack_transit_node(G, partition, proc)
        self.assertEqual(partition, [0, 0, 0, 0, 0, 1], 'Не то разбиение после запаковки')

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[5]['weight'], 5, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 4.5, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node6(self):
        G = nx.MultiDiGraph()

        edges = [(0, 1), (0, 2), (3, 2), (1, 7), (2, 4), (4, 7), (4, 5), (4, 6), (7, 5), (5, 6), (7, 6)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 0, 0, 0, 1, 1, 1, 1]

        f_val_1 = f_new(G, PG, partition)

        proc = 0
        pack_transit_node(G, partition, proc)

        proc = 1
        pack_transit_node(G, partition, proc)

        self.assertEqual(partition, [0, 1], 'Не то разбиение после запаковки')
        # print(G.nodes(data=True)[0])
        # print('-0-', [G.nodes[0]['inner_graph'].nodes[node]['initial_id'] for node in G.nodes[0]['inner_graph'].nodes])
        # print(G.nodes(data=True)[1])
        # print('-1-', [G.nodes[1]['inner_graph'].nodes[node]['initial_id'] for node in G.nodes[1]['inner_graph'].nodes])

        f_val_2 = f_new(G, PG, partition)

        self.assertEqual(G.nodes[0]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
        self.assertEqual(G.nodes[1]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5.5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 5.5, 'Неправильное значение длительности критического пути')

    # def test_f_new_with_transit_node7(self):
    #     G = nx.MultiDiGraph()

    #     edges = [(0, 1), (0, 2), (3, 2), (1, 7), (2, 4), (4, 7), (4, 5), (4, 6), (7, 5), (5, 6), (7, 6), (8, 0)]
    #     for u, v in edges:
    #         G.add_edge(u, v, weight=1, initial_edge=(u, v))

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1
    #         G.nodes[i]['initial_id'] = i
    #         G.nodes[i]['isTransit'] = False

    #     PG = input_graph('./data/processor_sets/4_1x1.txt')
    #     partition = [0, 0, 0, 0, 1, 1, 1, 1, 0]

    #     f_val_1 = f_new(G, PG, partition)
        

    #     partition[-1] = 2
    #     proc = 0
    #     pack_transit_node(G, partition, proc)
    #     partition[partition.index(2)] = 0
        
    #     proc = 1
    #     pack_transit_node(G, partition, proc)

    #     self.assertEqual(partition, [0, 0, 1], 'Не то разбиение после запаковки')
    #     # print(G.nodes(data=True)[1])
    #     # print(G.nodes(data=True)[2])

    #     f_val_2 = f_new(G, PG, partition)


    #     self.assertEqual(G.nodes[0]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')
    #     self.assertEqual(G.nodes[1]['weight'], 4, 'Не тот вес транзитной ноды (не совпадает с суммой весов внутренних вершин)')

    #     self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
    #     self.assertEqual(f_val_1, 5.5, 'Неправильное значение ЦФ')
    #     cp = findLongestPath(G, PG, partition)
    #     self.assertEqual(cp, 5.5, 'Неправильное значение длительности критического пути')
