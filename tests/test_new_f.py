import unittest

from algo.helpers import f_new, input_graph, create_transit_graph, findLongestPath

import networkx as nx


class TestNewF(unittest.TestCase):

    def test_f_new_with_transit_node1(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (1, 2), (2, 3)]

        for edge in edges:
            G.add_edge(*edge, weight=1, initial_edge=edge)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node2(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node3(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7), (2, 6), (3, 6)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node4(self):
        G = nx.DiGraph()

        edges = [(0, 1,), (0, 4), (1, 2), (1, 3), (2, 5), (3, 5), (5, 6), (4, 7), (6, 7), (2, 6), (3, 6), (8, 3), (9, 8)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 1, 0, 1, 0, 0, 0, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')

    def test_f_new_with_transit_node5(self):
        G = nx.DiGraph()

        edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 4.5, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node6(self):
        G = nx.DiGraph()

        edges = [(0, 1), (0, 2), (3, 2), (1, 7), (2, 4), (4, 7), (4, 5), (4, 6), (7, 5), (5, 6), (7, 6)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 0, 0, 0, 1, 1, 1, 1]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5.5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 5.5, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node7(self):
        G = nx.DiGraph()

        edges = [(0, 1), (0, 2), (3, 2), (1, 7), (2, 4), (4, 7), (4, 5), (4, 6), (7, 5), (5, 6), (7, 6), (8, 0)]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 0, 0, 0, 1, 1, 1, 1, 0]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5.75, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 5.75, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node8(self):
        G = nx.DiGraph()

        edges = [
            (0, 1), (2, 3),
            (4, 5), (6, 7),
            (8, 9), (10, 11),
            (12, 13), (14, 15),

            (1, 4), (3, 7), 
            (1, 9), (3, 10),
            (5, 12), (7, 14),
            (9, 12), (11, 14),
        ]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x2.txt')
        partition = [
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        ]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 6.5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 6.5, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node9(self):
        G = nx.DiGraph()

        edges = [
            (0, 1), (2, 3),
            (4, 5), (6, 7),
            (8, 9), (10, 11),
            (12, 13), (14, 15),

            (1, 4), (3, 7), 
            (1, 9), (3, 10),
            (5, 12), (7, 14),
            (9, 12), (11, 14),
        ]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x2.txt')
        partition = [
            0, 0, 0, 0,
            1, 1, 1, 1,
            3, 3, 3, 3,
            2, 2, 2, 2,
        ]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 5, 'Неправильное значение длительности критического пути')

    def test_f_new_with_transit_node10(self):
        G = nx.DiGraph()

        edges = [
            (0, 1), (1, 3),
            (0, 2), (2, 3),
        ]
        for u, v in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [
            0, 0, 1, 0,
        ]

        f_val_1 = f_new(G, PG, partition)

        transit_graph, transit_graph_data, node2subgraph, transit_partition = create_transit_graph(G, partition)
        f_val_2 = f_new(transit_graph, PG, transit_partition, transit_graph_data, node2subgraph)

        self.assertEqual(f_val_1, f_val_2, 'Значения ЦФ до и после запаковки не совпадают')
        self.assertEqual(f_val_1, 3.5, 'Неправильное значение ЦФ')
        cp = findLongestPath(G, PG, partition)
        self.assertEqual(cp, 3.5, 'Неправильное значение длительности критического пути')
