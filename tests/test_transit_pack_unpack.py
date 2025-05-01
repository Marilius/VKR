import unittest

from algo.helpers import pack_transit_node, unpack_transit_node

import networkx as nx
from copy import deepcopy


class TestTransitPackUnpack(unittest.TestCase):

    def test1(self):
        G = nx.MultiDiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)], weight=1)

        for i in range(len(G.nodes)):
            G.nodes[i]['weight'] = 1
            G.nodes[i]['isTransit'] = False
            G.nodes[i]['initial_id'] = i

        partition_initial = [0, 0, 1, 1]
        partition = deepcopy(partition_initial)

        pack_transit_node(G, partition, 0)
        pack_transit_node(G, partition, 1)

        partition = unpack_transit_node(G, partition)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test2(self):
        G = nx.MultiDiGraph()

        partition_initial = []
        for i in range(5):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i )

        partition = deepcopy(partition_initial)

        for proc in range(5):
            pack_transit_node(G, partition, proc)

        partition = unpack_transit_node(G, partition)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test3(self):
        G = nx.MultiDiGraph()

        partition_initial = []
        for i in range(10):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i % 2 )

        partition = deepcopy(partition_initial)
        
        for proc in range(5):
            pack_transit_node(G, partition, proc)

        partition = unpack_transit_node(G, partition)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test4(self):
        G = nx.MultiDiGraph()

        partition_initial = []
        for i in range(20):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i % 5 )

        partition = deepcopy(partition_initial)

        for proc in range(5):
            pack_transit_node(G, partition, proc)
            print(sorted(list(G.nodes)))

        partition = unpack_transit_node(G, partition)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')

    def test5(self):
        G = nx.MultiDiGraph()
        
        partition_initial = []
        for i in range(100):
            G.add_node(
                i,
                weight=1,
                isTransit=False,
                initial_id=i,
            )
            partition_initial.append( i % 10 )

        partition = deepcopy(partition_initial)
        
        for proc in range(10):
            pack_transit_node(G, partition, proc)
        
        partition = unpack_transit_node(G, partition)

        self.assertEqual(partition_initial, partition, 'разбиения до и после распаковки не совпадают')
