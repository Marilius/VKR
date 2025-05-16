import networkx as nx
import typing

from .basics import longest_paths_from_source

# TODO убрать рекурсию 


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
                # print('in_node:', in_node, 'out_node:', out_node, 'distance:', distances[out_node])
                transit_graph.add_edge(in_node, out_node, weight=distances[out_node])

    for node in transit_graph:
        transit_graph.nodes[node]['weight'] = G.nodes[node]['weight']
        transit_graph.nodes[node]['initial_node'] = node

    mapping: dict[int, int] = dict(zip(sorted(list(transit_graph.nodes)), range(len(transit_graph.nodes))))
    # print('mapping', mapping)

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

    # print('transit_graph', transit_graph)
    # print()
    transit_partition: list[int] = list(range(n))
    # for node in transit_graph:
        # transit_partition[node] = node2subgraph[node

    return (transit_graph, subgraphs_data, node2subgraph, transit_partition)


def pack_transit_node(G: nx.MultiDiGraph, partition: list[int], proc: int) -> None:
    # TODO распаковка внутренних транзитных узлов? 

    weight: int = 0
    for i in G.nodes:
        if partition[i] == proc:
            
            weight += G.nodes[i]['weight']

    # cur/init
    in_nodes: set[tuple[int, int]] = set()
    out_nodes: set[tuple[int, int]] = set()
    
    # cur [i, j, k] + w + initial_edge (tuple[int, int])
    in_edges: set[tuple[int, int, int, int, tuple[int, int]]] = set()
    out_edges: set[tuple[int, int, int, int, tuple[int, int]]] = set()

    adj: dict[int, dict[int, dict[int, dict[str, typing.Any]]]] = {node:nbrsdict for (node, nbrsdict) in G.adjacency()}
    # print(adj)

    for i in adj:
        if partition[i] == proc:
            for j in adj[i]:
                if partition[j] != proc:
                    for k in adj[i][j]:
                        # print('--->', j)
                        # out_nodes.add(i)
                        initial_edge: tuple[int, int] = adj[i][j][k]['initial_edge']
                        out_edges.add((i, j, k, adj[i][j][k]['weight'], initial_edge))
                        # out_edges.add((*initial_edge, k, adj[i][j][k]['weight']))
                        out_nodes.add((i, initial_edge[0]))
        else:
            for j in adj[i]:
                if partition[j] == proc:
                    for k in adj[i][j]:
                        # in_nodes.add(j)
                        initial_edge: tuple[int, int] = adj[i][j][k]['initial_edge']
                        in_edges.add((i, j, k, adj[i][j][k]['weight'], initial_edge))
                        # in_edges.add((*initial_edge, k, adj[i][j][k]['weight']))
                        in_nodes.add((j, initial_edge[1]))
    # # возможно, потом нужно переписать на мультиграф
    inner_graph: nx.DiGraph = nx.DiGraph()
    for u, v, key in G.edges(keys=True):
        if partition[u] == partition[v] == proc:
            inner_graph.add_edge(u, v, weight=G.edges[u, v, key]['weight'], initial_edge=G.edges[u, v, key]['initial_edge'])

    for node in G.nodes:
        if partition[node] == proc:
            node_w = G.nodes[node]['weight']

            initial_id = G.nodes[node]['initial_id']

            if node not in inner_graph:
                inner_graph.add_node(
                    node,
                    weight=node_w,
                    isTransit=False,
                    initial_id=initial_id,
                )
            else:
                inner_graph.nodes[node]['weight'] = node_w
                inner_graph.nodes[node]['isTransit'] = False
                inner_graph.nodes[node]['initial_id'] = initial_id

    paths: dict[tuple[int, int], int] = {}
    for i, init_i in in_nodes:
        distances = longest_paths_from_source(inner_graph, i)
        paths[(G.nodes[i]['initial_id'], -1)] = max(distances.values())
        for j, init_j in out_nodes:
            if distances[j] != -1:
                paths[(init_i, init_j)] = distances[j]

    n = len(G.nodes)
    G.add_node(
        n,
        weight=weight,
        in_nodes=in_nodes,
        out_nodes=out_nodes,
        in_edges=in_edges,
        out_edges=out_edges,
        paths=paths,
        inner_graph=inner_graph,
        isTransit=True,
    )

    for (u, v, k, w, initial_edge) in in_edges:
        # G.add_edge(u, n, weight=w, prev_in_node=v, initial_edge=adj[u][v][k]['initial_edge'])
        G.add_edge(u, n, weight=w, prev_in_node=v, initial_edge=initial_edge)

    for (u, v, k, w, initial_edge) in out_edges:
        # G.add_edge(n, v, weight=w, prev_out_node=u, initial_edge=adj[u][v][k]['initial_edge'])
        G.add_edge(n, v, weight=w, prev_out_node=u, initial_edge=initial_edge)

    i = 0
    node = 0
    while i < len(partition):
        if partition[i] == proc:
            partition.pop(i)
            G.remove_node(node)
        else:
            i += 1
        node += 1

    partition.append(proc)

    mapping: dict[int, int] = dict(zip(sorted(list(G.nodes)), range(len(G.nodes))))
    nx.relabel_nodes(G, mapping, copy=False)
    # for prev_label, new_label in mapping.items():
    #     if 'prev_labels' in G.nodes[new_label]:
    #         G.nodes[new_label]['prev_labels'].append(prev_label)
    #     else:
    #         G.nodes[new_label]['prev_labels'] = [prev_label]
    # TODO

def unpack_transit_partition(G: nx.DiGraph, partition: list[int]) -> list[int]:
    unpacked_partition_d: dict[int, int] = dict()

    for node in G.nodes:
        if G.nodes[node]['isTransit']:
            inner_graph: nx.DiGraph = G.nodes[node]['inner_graph']
            for inner_node in inner_graph.nodes:
                initial_id = inner_graph.nodes[inner_node]['initial_id']
                unpacked_partition_d[initial_id] = partition[node]
        else:
            initial_id = G.nodes[node]['initial_id']
            unpacked_partition_d[initial_id] = partition[node]

    unpacked_partition: list[int] = [0] * len(unpacked_partition_d)
    for k, v in unpacked_partition_d.items():
        unpacked_partition[k] = v

    return unpacked_partition
