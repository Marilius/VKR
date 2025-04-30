import settings

import networkx as nx

from os.path import isfile

from time import sleep


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
            inner_graph.nodes[node]['weight'] = G.nodes[node]['weight']

    paths: dict[tuple[int, int], int] = {}
    for i in in_nodes:
        distances = longest_paths_from_source(inner_graph, i)
        for j in out_nodes:
            if distances[j] != -1:
                paths[(i, j)] = distances[j]

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

    mapping: dict[int, int] = dict(zip(G.nodes, range(len(G.nodes))))
    nx.relabel_nodes(G, mapping, copy=False)
    for prev_label, new_label in mapping.items():
        if 'prev_labels' in G.nodes[new_label]:
            G.nodes[new_label]['prev_labels'].append(prev_label)
        else:
            G.nodes[new_label]['prev_labels'] = [prev_label]
    # TODO
    
def unpack_transit_node(G: nx.DiGraph, partition: list[int], proc: int) -> None:
    ...

def longest_paths_from_source(G: nx.DiGraph, source: int) -> dict[int, int]:
    # TODO именно пути?
    stack: list[int] = []
    adj: dict[int, list[int]] = {node:list(children.keys()) for (node, children) in G.adjacency()}

    dist = { node:-1 for node in G.nodes }

    stack.append(source)
    dist[source] = 0

    while stack: 
        u = stack.pop()

        if dist[u] != -1: 
            for i in adj[u]: 
                if (dist[i] < dist[u] + G.nodes[i]['weight']): 
                    dist[i] = dist[u] + G.nodes[i]['weight']
                    stack.append(i)

    return dist

def check_cut_ratio(G: nx.Graph | None, partition: list[int] | None, cr_max: float) -> bool:
    """
    Checks if the cut ratio of a given graph with a given partition is not more than a given value.

    Args:
        G (nx.Graph | None): The graph to calculate the cut ratio for.
        partition (list[int] | None): A list of the partition assignment for each node.
        cr_max (float): The maximum allowed cut ratio.

    Returns:
        bool: Whether the cut ratio is not more than cr_max.
    """

    if G is None or partition is None:
        return False

    return calc_cut_ratio(G, partition) <= cr_max  # type: ignore

def dfs(
        node: int,
        G: nx.MultiDiGraph,
        PG: nx.Graph,
        partition: list[int],
        dp: list[float],
        vis: list[bool],
        best_paths: list[list[int]],
        adj: dict[int, dict[int, dict[int, dict[str, int]]]],
    ) -> None:
    # Mark as visited 
    vis[node] = True
    # adj:  = [list(children.keys()) for node, children in G.adjacency()]
    # Traverse for all its children 
    for child in adj[node]:
        node_w = G.nodes[node]['weight'] / PG.nodes[partition[node]]['weight']
        if G.nodes[node]['isTransit']:  # TODO
            print(G.nodes[node]['paths'])
            print(node, child)
            node_w = G.nodes[node]['paths'][()] # нужен не вес самой ноды, а самого длинного пути внутри
        if not vis[child]:
            # If not visited 
            dfs(child, G, PG, partition, dp, vis, best_paths, adj)

        for edge_key in adj[node][child]:
            # Store the max of the paths
            edge_w = 0 if partition[node] == partition[child] else G.get_edge_data(node, child, edge_key)["weight"]
            new_val = dp[child] + node_w + edge_w

            if new_val > dp[node]:
                dp[node] = new_val
                best_paths[node] = [node] + best_paths[child]  # Обновляем путь

def findLongestPath(G: nx.MultiDiGraph, PG: nx.Graph, partition: list[int]) -> tuple[float, list[int]]:
    n = len(G.nodes)

    # Dp array
    dp = [G.nodes[i]['weight']/PG.nodes[partition[i]]['weight'] for i in range(n)]
    best_paths: list[list[int]] = [[node] for node in G.nodes]  # Изначально путь до каждой вершины — только она сама
    adj: dict[int, dict[int, dict[int, dict[str, int]]]] = {node:nbrsdict for (node, nbrsdict) in G.adjacency()}

    # Visited array to know if the node 
    # has been visited previously or not 
    vis = [False] * n

    # Call DFS for every unvisited vertex 
    for i in range(0, n):
        if not vis[i]:
            dfs(i, G, PG, partition, dp, vis, best_paths, adj)

    max_val = max(dp)
    max_idx = dp.index(max_val)

    return max_val, best_paths[max_idx]

def f_new(G: nx.MultiDiGraph, PG: nx.Graph, partition: list[int] | None) -> float:
    if partition is None:
        return 2 * settings.BIG_NUMBER

    p_loads = [0] * len(PG)

    for i in range(len(partition)):
        p_loads[partition[i]] += G.nodes[i]['weight']

    for i in range(len(PG)):
        p_loads[i] /= PG.nodes[i]['weight'] 

    cp_length, cp = findLongestPath(G, PG, partition)

    return max(*p_loads, cp_length)

def f(G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> float:
    """
    Calculates the objective function value.

    Args:
        G (nx.Graph | None): The graph to calculate the load for.
        PG (nx.Graph): The processor graph.
        partition (list[int] | None): A list of the partition assignment for each node.
        cr_max (float): The maximum allowed cut ratio.

    Returns:
        float: Objective function value.
    """
    p_loads = [0] * len(PG)

    if G is None or partition is None:
        return 2 * settings.BIG_NUMBER

    for i in range(len(partition)):
        p_loads[partition[i]] += G.nodes[i]['weight']

    for i in range(len(PG)):
        p_loads[i] /= PG.nodes[i]['weight'] 

    penalty = 0 if check_cut_ratio(G, partition, cr_max) else settings.BIG_NUMBER

    return max(p_loads) + penalty

def calc_edgecut(G: nx.Graph, partition: list[int]) -> int:
    """
    Calculate the number of edges crossing between parts in a graph partition.

    Args:
        G (nx.Graph): The graph to calculate the edgecut for.
        partition (list[int]): The partition of the graph.

    Returns:
        int: The number of edges crossing between different parts.
    """
    edgecut: int = 0
    for edge in G.edges:
        node1, node2 = edge
        if partition[node1] != partition[node2]:
            edgecut += 1

    return edgecut

def calc_cut_ratio(G: nx.Graph | None, partition: list[int] | None) -> float | None:
    """
    Calculate the cut ratio of a graph given a partitioning of the graph.

    The cut ratio is the number of edges crossing between different partitions
    divided by the total number of edges in the graph.

    Args:
        G (nx.Graph): The graph to calculate the cut ratio for.
        partition (list[int]): The partitioning of the graph.

    Returns:
        float | None: The cut ratio of the graph given the partitioning. If the
            graph or partition is None, returns None.
    """

    if G is None or partition is None:
        return None

    if len(G.edges) == 0:
        return 0

    return calc_edgecut(G, partition) / len(G.edges)

def unpack_mk(mk_partition: list[int], mk_data: list[int]) -> list[int]:
    """
    Unpacks a coarsened partition into its original form using mapping data.

    Args:
        mk_partition (list[int]): A list representing the coarsened partition,
            where each index corresponds to a coarsened node and the value is
            the processor assigned to that node.
        mk_data (list[int]): A list representing the original nodes' mapping
            to coarsened nodes.

    Returns:
        list[int]: A list representing the original partition where each index
            corresponds to an original node and the value is the processor
            assigned to that node.
    """

    ans = mk_data.copy()
    mapping: dict[int, int] = dict()

    for mk_id, proc in enumerate(mk_partition):
        mapping[mk_id] = proc

    for i in range(len(mk_data)):
        ans[i] = mapping[ans[i]]

    return ans
