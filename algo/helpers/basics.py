import settings

import networkx as nx

import typing


def longest_paths_from_source(G: nx.DiGraph, source: int) -> dict[int, int]:
    # TODO именно пути?
    stack: list[int] = []
    adj: dict[int, list[int]] = {node:list(children.keys()) for (node, children) in G.adjacency()}

    dist = { node:-1 for node in G.nodes }

    stack.append(source)
    dist[source] = G.nodes[source]['weight']

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
        adj: dict[int, dict[int, dict[int, dict[str, typing.Any]]]],
    ) -> None:
    # adj:  = [list(children.keys()) for node, children in G.adjacency()]
    # Traverse for all its children 
    
    # Mark as visited 
    vis[node] = True
    
    for child in adj[node]:
            
        if not vis[child]:
            # If not visited 
            dfs(child, G, PG, partition, dp, vis, best_paths, adj)

        if G.nodes[node]['isTransit']:  # TODO
            print('paths', G.nodes[node]['paths'])
            print(node, child)
            
            for edge_key in adj[node][child]:
                initial_edge: tuple[int, int] = adj[node][child][edge_key]['initial_edge']
                print(node, child, edge_key, 'initial_edge', initial_edge, adj[node][child][edge_key])
                
                node_w = max(map(lambda x: x[1], filter(lambda x: x[0][1] == initial_edge[0], G.nodes[node]['paths'].items())))
                node_w = max(map(lambda x: x[1], filter(lambda x: x[0][1] == initial_edge[0], G.nodes[node]['paths'].items())))
                edge_w = 0 if partition[node] == partition[child] else G.get_edge_data(node, child, edge_key)["weight"]
                new_val = dp[child] + node_w + edge_w
                
                if new_val > dp[node]:
                    dp[node] = new_val
                    best_paths[node] = [node] + best_paths[child]  # Обновляем путь
        else:
            for edge_key in adj[node][child]:
                node_w = G.nodes[node]['weight'] / PG.nodes[partition[node]]['weight']
                # Store the max of the paths
                edge_w = 0 if partition[node] == partition[child] else G.get_edge_data(node, child, edge_key)["weight"]
                new_val = dp[child] + node_w + edge_w

                if new_val > dp[node]:
                    dp[node] = new_val
                    best_paths[node] = [node] + best_paths[child]  # Обновляем путь

def findLongestPath(G: nx.MultiDiGraph, PG: nx.Graph, partition: list[int]) -> tuple[float, list[int]]:
    n = len(G.nodes)

    # Dp array
    dp = [
        max(G.nodes[i]['paths'].values()) if G.nodes[i]['isTransit']
        else G.nodes[i]['weight']/PG.nodes[partition[i]]['weight']
        for i in range(n)
    ]
    best_paths: list[list[int]] = [[node] for node in G.nodes]  # Изначально путь до каждой вершины — только она сама
    adj: dict[int, dict[int, dict[int, dict[str, int]]]] = {node:nbrsdict for (node, nbrsdict) in G.adjacency()}

    # Visited array to know if the node 
    # has been visited previously or not 
    vis = [False] * n

    # Call DFS for every unvisited vertex 
    for i in range(0, n):
        if not vis[i]:
            dfs(i, G, PG, partition, dp, vis, best_paths, adj)
            print('dp', dp)

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
