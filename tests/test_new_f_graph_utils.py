import unittest

from algo.helpers import input_graph, findLongestPath, longest_paths_from_source

import networkx as nx


class TestNewF(unittest.TestCase):

    def test_cp_1(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [1, 1, 1, 1]

        cp_length = findLongestPath(G, PG, partition)
        self.assertEqual(cp_length, 4, 'Значение ЦФ не то')
        # self.assertEqual(cp, [0, 1, 2, 3], 'Не тот критический путь')

    def test_cp_2(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1,), (0, 2), (2, 1), (1, 3), (2, 3)], weight=1)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['isTransit'] = False

        PG = input_graph('./data/processor_sets/4_1x1.txt')
        partition = [0, 1, 0, 1]

        cp_length = findLongestPath(G, PG, partition)
        self.assertEqual(cp_length, 3.5, 'Значение ЦФ не то')
        # self.assertEqual(cp, [0, 2, 1, 3], 'Не тот критический путь')

    # def test_longest_path_from_source1(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1

    #     source = 0
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, {0:1, 1:2, 2:3, 3:4}, 'Неправильнык длины путей')

    # def test_longest_path_from_source2(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1
    #         G.nodes[i]['isTransit'] = False

    #     source = 1
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, {0:-1, 1:1, 2:2, 3:3}, 'Неправильные длины путей')
        
    # def test_longest_path_from_source3(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1
    #         G.nodes[i]['isTransit'] = False

    #     source = 2
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, {0:-1, 1:-1, 2:1, 3:2}, 'Неправильные длины путей')
        
    # def test_longest_path_from_source4(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from([(0, 1,), (1, 2), (2, 3)], weight=1)

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1
    #         G.nodes[i]['isTransit'] = False

    #     source = 3
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, { 0:-1, 1:-1, 2:-1, 3:1 }, 'Неправильные длины путей')

    # def test_longest_path_from_source5(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from(
    #         [
    #             (1, 3),
    #             (3, 4),
    #             (3, 7),
    #             (4, 5),
    #             (4, 6),
    #             (6, 7),
    #             (1, 5),
    #             (1, 2),
    #             (0, 1),
    #         ],
    #         weight=1,
    #     )

    #     for i in range(len(G.nodes)):
    #         G.nodes[i]['weight'] = 1
    #         G.nodes[i]['isTransit'] = False

    #     source = 0
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, { 0:1, 1:2, 2:3, 3:3, 4:4, 5:5, 6:5, 7:6 }, 'Неправильные длины путей')

    # def test_longest_path_from_source6(self):
    #     G = nx.DiGraph()
    #     G.add_edges_from(
    #         [
    #             (1, 3),
    #             (3, 4),
    #             (3, 7),
    #             (4, 5),
    #             (4, 6),
    #             (6, 7),
    #             (1, 5),
    #             (1, 2),
    #             (0, 1),
    #         ],
    #         weight=1,
    #     )

    #     for w, i in enumerate(range(len(G.nodes))):
    #         G.nodes[i]['weight'] = w
    #         G.nodes[i]['isTransit'] = False

    #     source = 0
    #     longest_paths = longest_paths_from_source(G, source)
    #     self.assertEqual(longest_paths, {0:0, 1:1, 2:3, 3:4, 4:8, 5:13, 6:14, 7:21}, 'Неправильные длины путей')
