from .basics import check_cut_ratio, unpack_mk

import settings

import networkx as nx

from os import makedirs
from os.path import isfile

from functools import wraps

from time import sleep


def get_cache_filename(func_name: str, **kwargs) -> str:
    file_name = []
    for arg in kwargs.values():
        if isinstance(arg, nx.Graph):
            node_attr = 'weight' if 'node_weight_attr' in arg.graph else None
            file_name.append(nx.weisfeiler_lehman_graph_hash(arg, node_attr=node_attr))
        else:
            file_name.append(arg)
    path = f'{settings.CACHE_DIR}/{func_name}/{"_".join(file_name)}.txt'
    return path

def load_cache(func_name: str, *args, **kwargs) -> list[int] | None:
    path = get_cache_filename(func_name, **kwargs)
    if isfile(path):
        with open(path, 'r') as f:
            line = f.readline()
            partition = list(map(int, line.split()))
            return partition

    return None

def write_cache(func_name: str, partition: list[int] | None, **kwargs) -> None:
    path = get_cache_filename(func_name, **kwargs)

    makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    with open(path, 'w') as file:
        if partition:
            file.write(' '.join(map(str, partition)))
        else:
            file.write('None')

def add_cache_check(func):
    def f(check_cache: bool):
        @wraps(func)
        def wrapped(**kwargs) -> list[int] | None:
            if not kwargs: 
                raise TypeError('No kwargs given to function')
            if check_cache:
                partition = load_cache(func.__name__, **kwargs)
                G = kwargs['G']
                cr_max = kwargs['cr_max']
                if partition and len(partition) == len(G.nodes) and check_cut_ratio(G, partition, cr_max):
                    return partition

            partition = func(**kwargs)

            if check_cache:
                write_cache(func.__name__, partition, **kwargs)

            return partition

        return wrapped
    return f

def input_graph(path: str) -> nx.DiGraph:
    """
    Reads a graph from a file in either the format used by the task_graph_generator.py script or the
    "node_id node_weight child1 child2 ..." format. The graph is read into a NetworkX graph object.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.Graph: the read graph
    """
    if 'gen_data' in path:
        graph, _, _ = input_generated_graph_and_processors_from_file(path)
    else:
        graph = input_networkx_graph_from_file(path)

    return graph

def input_networkx_graph_from_file(path: str) -> nx.DiGraph:
    """
    Reads a graph from a file in the "node_id node_weight child1 child2 ..." format. The graph is read into a NetworkX graph object.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.DiGraph: the read graph
    """
    G = nx.DiGraph()

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
            for child in children:
                G.add_edge(name, child, weight=1)
                # TODO
            # G.add_edges_from((name, child) for )
    G.graph['node_weight_attr'] = 'weight'

    mk = 'mk_' if 'data_mk' in path else ''

    graph_name = mk + path.split('/')[-1].split('.')[0]
    G.graph['graph_name'] = graph_name

    return G

def input_networkx_unweighted_graph_from_file(path: str) -> nx.Graph:
    """
    Reads an unweighted graph from a file in format
    "node_id node_weight child1 child2 ...". 
    The graph is read into a NetworkX graph object, and the node weights are set to 1.

    Args:
        path (str): path to the file containing the graph

    Returns:
        nx.Graph: the read graph
    """
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
    """
    Reads a partition from a file created by task_graph_generator.py.

    Args:
        path (str): The path to the file containing the partition.

    Returns:
        list[int]: The partition read from the file.

    Raises:
        FileNotFoundError: If the file specified by path does not exist.
    """
    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    exact_partition: list[int] = []

    with open(path, 'r') as f:
        exact_partition = list(map(int, f.readline().strip().split()))

    return exact_partition

def input_generated_graph_and_processors_from_file(path: str) -> tuple[nx.DiGraph, list[int], dict[str, int|list[int]]]:
    """
    Reads a graph and processor details from a file created by task_graph_generator.py.

    Args:
        path (str): The path to the file containing the graph and processor information.

    Returns:
        tuple[nx.Graph, list[int], dict[str, int|list[int]]]: A tuple containing the graph as a 
        NetworkX object, a list of processors, and a dictionary of parameters including 'p', 'L', 
        'min_l', 'max_l', 'N_e', and 'N_s'.

    Raises:
        FileNotFoundError: If the file specified by path does not exist.
    """

    if not isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    G = nx.Graph()
    params: dict[str, int|list[int]] = {}

    with open(path, 'r') as f:
        print(path)
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

def do_unpack_mk(mk_partition: list[int], mk_data_path: str) -> list[int]:
    """
    Unpacks a coarsened partition into its original form using mapping data.

    Args:
        mk_partition (list[int]): A list representing the coarsened partition,
            where each index corresponds to a coarsened node and the value is
            the processor assigned to that node.
        mk_data_path (str): The path to the mapping data file.

    Returns:
        list[int]: A list representing the original partition where each index
            corresponds to an original node and the value is the processor
            assigned to that node.
    """
    while not isfile(mk_data_path):
        print('waiting for: ', mk_data_path)
        sleep(10)

    with open(mk_data_path, 'r') as file:
        line = file.readline()
        mk_data = list(map(int, line.split()))

        return unpack_mk(mk_partition, mk_data)

def fix_rand_graph_file(path: str) -> None:
    """
    Fixes the node identifiers in a graph file by ensuring they are contiguous
    and starting from 0. Reads the graph from the specified path, renames the
    node identifiers in the file, and writes the corrected graph back to the
    file.

    Args:
        path (str): The file path to the graph file that needs to be fixed.

    Raises:
        AssertionError: If the graph nodes are not contiguous and starting from 0
            after the fix.
    """

    graph: nx.Graph = input_graph(path)

    with open(path, 'r') as file:
        filedata = file.read()
        for old_id, new_id in zip(list(sorted(list(graph.nodes))), list(range(len(graph.nodes)))):
            filedata = filedata.replace(f'\t{old_id}\n', f'\t{new_id}\n')
            filedata = filedata.replace(f'\t{old_id}\t', f'\t{new_id}\t')
            filedata = filedata.replace(f'\n{old_id}\t', f'\n{new_id}\t')
            filedata = filedata.replace(f'\n{old_id}\n', f'\n{new_id}\n')

    # Write the file out again
    with open(path, 'w') as file:
        file.write(filedata)

    graph: nx.Graph = input_graph(path)
    assert list(sorted(list(graph.nodes))) == list(range(len(graph.nodes)))
