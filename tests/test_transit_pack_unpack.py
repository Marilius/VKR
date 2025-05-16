import unittest

from algo.helpers import create_transit_graph, unpack_transit_partition

import networkx as nx


class TestTransitPackUnpack(unittest.TestCase):

    def test1(self):
        G = nx.DiGraph()

        edges = [(0, 1), (1, 2), (2, 3)]
        for (u, v) in edges:
            G.add_edge(u, v, weight=1, initial_edge=(u, v))

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['initial_id'] = i

        partition_initial = [0, 0, 1, 1]

        _, transit_graph_data, _, transit_partition = create_transit_graph(G, partition_initial)
        partition = unpack_transit_partition(transit_partition, transit_graph_data)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test2(self):
        G = nx.DiGraph()

        partition_initial = []
        for i in range(5):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i )

        _, transit_graph_data, _, transit_partition = create_transit_graph(G, partition_initial)
        partition = unpack_transit_partition(transit_partition, transit_graph_data)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test3(self):
        G = nx.DiGraph()

        partition_initial = []
        for i in range(10):
            G.add_node(
                i,
                weight=1,
                initial_id=i,
            )
            partition_initial.append( i % 2 )
        
        _, transit_graph_data, _, transit_partition = create_transit_graph(G, partition_initial)
        partition = unpack_transit_partition(transit_partition, transit_graph_data)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test4(self):
        G = nx.DiGraph()

        partition_initial = []
        for i in range(20):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i % 5 )

        _, transit_graph_data, _, transit_partition = create_transit_graph(G, partition_initial)
        partition = unpack_transit_partition(transit_partition, transit_graph_data)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test5(self):
        G = nx.DiGraph()
        
        partition_initial = []
        for i in range(100):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i % 10 )

        _, transit_graph_data, _, transit_partition = create_transit_graph(G, partition_initial)
        partition = unpack_transit_partition(transit_partition, transit_graph_data)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')
