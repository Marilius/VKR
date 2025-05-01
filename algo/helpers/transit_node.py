import networkx as nx


from .basics import longest_paths_from_source

# TODO убрать рекурсию 


def pack_transit_node(G: nx.MultiDiGraph, partition: list[int], proc: int) -> None:
    # TODO распаковка внутренних транзитных узлов? 

    weight: int = 0
    for i in G.nodes:
        if partition[i] == proc:
            weight += G.nodes[i]['weight']

    in_nodes: set[int] = set()
    out_nodes: set[int] = set()
    in_edges: set[tuple[int, int, int, int]] = set()
    out_edges: set[tuple[int, int, int, int]] = set()

    adj: dict[int, dict[int, dict[int, dict[str, int]]]] = {node:nbrsdict for (node, nbrsdict) in G.adjacency()}

    # for i in G.nodes:
    for i in adj:
        if partition[i] == proc:
            for j in adj[i]:
                if partition[j] != proc:
                    for k in adj[i][j]:
                        # print('--->', j)
                        out_edges.add((i, j, k, adj[i][j][k]['weight']))
                        out_nodes.add(i)
        else:
            for j in adj[i]:
                if partition[j] == proc:
                    for k in adj[i][j]:
                        in_edges.add((i, j, k, adj[i][j][k]['weight']))
                        in_nodes.add(j)

    # возможно, потом нужно переписать на мультиграф
    inner_graph: nx.DiGraph = nx.DiGraph()
    for u, v, key, w in G.edges(data='weight', keys=True):
        # print('-->', u, v, key, weight)
        if partition[u] == partition[v] == proc:
            inner_graph.add_edge(u, v, weight=G.edges[u, v, key]['weight'])

    for node in G.nodes:
        if partition[node] == proc:
            print(G.nodes(data=True)[node])
            node_w = G.nodes[node]['weight']
            
            if 'initial_id' not in G.nodes(data=True)[node]:
                print('--->', G.nodes(data=True)[node])
                print('<--->', partition, '---', len(partition), '---', proc)
                print('<--->', sorted(list(G.nodes)))
                print('node: ', node, 'proc:', partition[node])
                # raise
            
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
    for i in in_nodes:
        distances = longest_paths_from_source(inner_graph, i)
        for j in out_nodes:
            if distances[j] != -1:
                paths[(i, j)] = distances[j]

    n = len(G.nodes)
    print('ADDING', n)
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

    for (u, v, k, w) in in_edges:
        G.add_edge(u, n, k, weight=w, prev_in_node=v)

    for (u, v, k, w) in out_edges:
        G.add_edge(n, v, k, weight=w, prev_out_node=u)
        # G.add_edge(n, v, weight=G.edges[u, v]['weight'], prev_out_node=u)

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
    for prev_label, new_label in mapping.items():
        if 'prev_labels' in G.nodes[new_label]:
            G.nodes[new_label]['prev_labels'].append(prev_label)
        else:
            G.nodes[new_label]['prev_labels'] = [prev_label]
    # TODO
    
def unpack_transit_node(G: nx.DiGraph, partition: list[int]) -> list[int]:
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
