from .basics import longest_paths_from_source

import networkx as nx


def create_transit_graph(G: nx.DiGraph, partition: list[int]) -> tuple[nx.DiGraph, dict[int, tuple[set[int], int, set[int]]], dict[int, int], list[int]]:
    transit_graph: nx.DiGraph = nx.DiGraph()

    n: int = len(set(partition))
    nodes_in_subgraphs: list[set[int]] = [set() for _ in range(n)]

    for i, subgraph in enumerate(partition):
        nodes_in_subgraphs[subgraph].add(i)

    for i, nodes in enumerate(nodes_in_subgraphs):
        inner_graph: nx.DiGraph = nx.DiGraph()
        for node in nodes:
            inner_graph.add_node(node, weight=G.nodes[node]['weight'])

        for u in nodes:
            for v in G[u]:
                if partition[u] == partition[v]:
                    inner_graph.add_edge(u, v, weight=G.edges[u, v]['weight'])

        in_nodes: set = set()
        out_nodes: set = set()
        for node in nodes:
            successors = list(G.successors(node))
            predecessors = list(G.predecessors(node))

            if not successors:
                out_nodes.add(node)
            if not predecessors:
                in_nodes.add(node)

            for s in successors:
                if partition[node] != partition[s]:
                    out_nodes.add(node)
                    transit_graph.add_edge(node, s, weight=G[node][s]['weight'])

            for p in predecessors:
                if partition[node] != partition[p]:
                    in_nodes.add(node)
                    transit_graph.add_edge(p, node, weight=G[p][node]['weight'])

        for in_node in in_nodes:
            distances = longest_paths_from_source(inner_graph, in_node)
            for out_node in out_nodes:
                if in_node == out_node or distances[out_node] < 0:
                    continue
                transit_graph.add_edge(in_node, out_node, weight=distances[out_node])

    for node in transit_graph:
        transit_graph.nodes[node]['weight'] = G.nodes[node]['weight']
        transit_graph.nodes[node]['initial_node'] = node

    mapping: dict[int, int] = dict(zip(sorted(list(transit_graph.nodes)), range(len(transit_graph.nodes))))

    subgraphs_data: dict[int, tuple[set[int], int, set[int]]] = {}
    transit_nodes = set(transit_graph)
    for i, nodes in enumerate(nodes_in_subgraphs):
        w_i: int = 0
        nodes_i_initial: set[int] = set()

        for node in nodes:
            w_i += G.nodes[node]['weight']
            nodes_i_initial.add(node)

        nodes_in_subgraph_initial = set(nodes).intersection(transit_nodes)
        nodes_in_subgraph_final: set[int] = set()
        for node in nodes_in_subgraph_initial:
            nodes_in_subgraph_final.add(mapping[node])
        
        subgraphs_data[i] = (
            nodes_in_subgraph_final, w_i, nodes_i_initial
        )

    node2subgraph: dict[int, int] = dict()
    for k, (nodes, _, _) in subgraphs_data.items():
        for node in nodes:
            node2subgraph[node] = k
    nx.relabel_nodes(transit_graph, mapping, copy=False)

    transit_partition: list[int] = list(range(n))

    return (transit_graph, subgraphs_data, node2subgraph, transit_partition)

def unpack_transit_partition(
    partition: list[int],
    subgraphs_data: dict[int, tuple[set[int], int, set[int]]],
) -> list[int]:
    unpacked_partition_d: dict[int, int] = dict()

    for i, (_, _, nodes_i_initial) in subgraphs_data.items():
        for node in nodes_i_initial:
            unpacked_partition_d[node] = partition[i]

    unpacked_partition: list[int] = [0] * len(unpacked_partition_d)
    for k, v in unpacked_partition_d.items():
        unpacked_partition[k] = v

    return unpacked_partition
