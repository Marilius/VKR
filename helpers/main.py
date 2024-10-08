import networkx as nx

from os.path import isfile

from time import sleep


def input_networkx_graph_from_file(path: str) -> nx.Graph:
    G = nx.Graph()

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)

        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, size, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=size)
            G.add_edges_from((name, child) for child in children)
    G.graph['node_weight_attr'] = 'weight'
    
    mk = 'mk_' if 'data_mk' in path else ''

    graph_name = mk + path.split('/')[-1].split('.')[0]
    G.graph['graph_name'] = graph_name

    return G


def input_networkx_unweighted_graph_from_file(path: str) -> nx.Graph:
    G = nx.Graph()

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)

        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, _, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=1)
            G.add_edges_from((name, child) for child in children)

    graph_name = path.split('/')[-1].split('.')[0]
    mk = 'mk_' if 'data_mk' in path else ''

    G.graph['graph_name'] = mk + graph_name
    
    return G


def input_generated_graph_partition(path: str) -> list[int]:
    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    exact_partition: list[int] = []

    with open(path, 'r') as f:
        exact_partition = list(map(int, f.readline().strip().split()))

    return exact_partition


def input_generated_graph_and_processors_from_file(path: str) -> tuple[nx.Graph, list[int], dict[str, int|list[int]]]:
    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')
        
    # FORMAT = "{p}\n{L}\n{min_l} {max_l}\n{N_e} {N_s}\n"

    G = nx.Graph()
    params: dict[str, int|list[int]] = {}

    with open(path, 'r') as f:
        p = list(map(int, f.readline().strip().split()))
        L = int(f.readline())
        min_l, max_l = list(map(int, f.readline().strip().split()))
        N_e, N_s = list(map(int, f.readline().strip().split()))

        params = {'p': p, 'L': L, 'min_l': min_l, 'max_l': max_l, 'N_e': N_e, 'N_s': N_s}

        for line in f.readlines():
            name, size, *children = map(int, line.strip().split())
            # name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=size)
            G.add_edges_from((name, child) for child in children)
    G.graph['node_weight_attr'] = 'weight'
    
    graph_name = path.split('/')[-1].split('.')[0]
    G.graph['graph_name'] = graph_name

    return G, p, params


def calc_edgecut(G: nx.Graph, partition: list[int]) -> int:
    edgecut = 0
    # print(partition)
    # print(G.nodes(data=True))
    for edge in G.edges:
        node1, node2 = edge
        if partition[node1] != partition[node2]:
            edgecut += 1

    return edgecut


def calc_cut_ratio(G: nx.Graph | None, partition: list[int] | None) -> float | None:
    if G is None or partition is None:
        return None

    if len(G.edges) == 0:
        return 0

    return calc_edgecut(G, partition) / len(G.edges)


def unpack_mk(mk_partition: list[int], mk_data: list[int]) -> list[int]:
    ans = mk_data.copy()
    mapping: dict[int, int] = dict()

    for mk_id, proc in enumerate(mk_partition):
        mapping[mk_id] = proc

    for i in range(len(mk_data)):
        ans[i] = mapping[ans[i]]

    return ans


def do_unpack_mk(mk_partition: list[int], mk_data_path: str) -> list[int]:
    while not isfile(mk_data_path):
        print('waiting for: ', mk_data_path)
        sleep(10)

    with open(mk_data_path, 'r') as file:
        line = file.readline()
        mk_data = list(map(int, line.split()))

        return unpack_mk(mk_partition, mk_data)
