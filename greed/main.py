#%%
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import networkx as nx
import metis
from copy import deepcopy 


CUT_RATIO = 0.7
PENALTY = True
BIG_NUMBER = 1e10

# For k-way clustering, the appropriate options are:

# objtype   = 'cut' or 'vol'
# ctype     = 'rm' or 'shem'
# iptype    = 'grow', 'random', 'edge', 'node'
# rtype     = 'fm', 'greedy', 'sep2sided', 'sep1sided'
# ncuts     = integer, number of cut attempts (default = 1)
# niter     = integer, number of iterations (default = 10)
# ufactor   = integer, maximum load imbalance of (1+x)/1000
# minconn   = bool, minimize degree of subdomain graph
# contig    = bool, force contiguous partitions
# seed      = integer, RNG seed
# numbering = 0 (C-style) or 1 (Fortran-style) indices
# dbglvl    = Debug flag bitfield


# G = nx.Graph()
# G.add_node(0)
# G.add_node(1, weight=2)
# G.add_node(2, weight=3)
# nx.draw(G, with_labels=True)
# plt.show()


# G = metis.example_networkx()
# (edgecuts, parts) = metis.part_graph(G, 3)
# colors = ['red','blue','green']
# for i, p in enumerate(parts):
#     G.node[i]['color'] = colors[p]


def input_networkx_graph_from_file(path: str) -> nx.Graph:
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, size, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=size)
            G.add_edges_from((name, child) for child in children)
    G.graph['node_weight_attr'] = 'weight'
    
    return G


def input_networkx_unweighted_graph_from_file(path: str) -> nx.Graph:
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, _, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=1)
            G.add_edges_from((name, child) for child in children)
    return G


def f(G: nx.Graph, PG: nx.Graph, partition: list[int]) -> float:
    p_loads = [0] * len(PG)
    for i in range(len(partition)):
        p_loads[partition[i]] += G.nodes[i]['weight']

    for i in range(len(PG)):
        p_loads[i] /= PG.nodes[i]['weight'] 
    
    penalty = 0

    if PENALTY:
        penalty = 0 if check_cut_ratio(G, partition) else BIG_NUMBER

    return max(p_loads) + penalty


def calc_edgecut(G: nx.Graph, partition: list[int]) -> int:
    edgecut = 0
    for edge in G.edges:
        node1, node2 = edge
        if partition[node1] != partition[node2]:
            edgecut += 1
    return edgecut


def calc_cut_ratio(G: nx.Graph, partition: list[int]) -> float:
    return calc_edgecut(G, partition) / len(G.edges)


def check_cut_ratio(G: nx.Graph, partition: list[int]) -> bool:
    return calc_cut_ratio(G, partition) <= CUT_RATIO


def postprocessing_phase(G: nx.Graph, PG: nx.Graph, partition: list[int]) -> list[int]:
    # The postprocessing phase has the following scheme.
    # 1) Select the most loaded processor; denote it as P1;
    # 2) For each task A assigned to P1, in decreasing order by task execution time:
    #   a) Choose the fastest processor P2 of the processors meeting the following
    #   constraint: if the task A is reassigned from P1 to P2, then
    #   max(load of P1, load of P2) decreases, and the CR constraint is met;
    #   b) If such processor P2 was found, reassign A from P1 to P2; go to step 1;
    #   else stop considering tasks on P1 with the same execution time as A until
    #   return to step 1;
    #   c) if last of execution times for tasks on P1 was discarded in step b, then stop.
    p_loads = [0] * len(PG)
    p_order = list(range(len(PG)))
    p_order.sort(key=lambda i: PG.nodes[i]['weight'], reverse=True)

    for i in range(len(partition)):
        p_loads[partition[i]] += G.nodes[i]['weight']

    flag = True
    while flag:

        p1 = None
        p1_time = 0
        for i, i_load in enumerate(p_loads):
            if i_load / PG.nodes[i]['weight'] > p1_time or p1 is None:
                p1 = i
                p1_time = i_load / PG.nodes[i]['weight']

        while flag:
            flag = False
            
            a = None
            a_weight = 0
            for job, proc in enumerate(partition):
                if proc == p1:
                    if G.nodes[job]['weight'] > a_weight:
                        a = job
                        a_weight = G.nodes[job]['weight']

            for proc in p_order:
                if proc != p1:
                    if max(p_loads[proc] / PG.nodes[proc]['weight'], p_loads[p1] / PG.nodes[p1]['weight']) \
                            > max((p_loads[p1] - a_weight)/ PG.nodes[p1]['weight'], (p_loads[proc] + a_weight) / PG.nodes[proc]['weight']):
                        partition_copy = deepcopy(partition)
                        partition_copy[a] = proc
                        if check_cut_ratio(G, partition_copy):
                            p_loads[proc] += a_weight
                            p_loads[p1] -= a_weight
                            partition[a] = proc
                            flag = True
                            break
            
            if flag:
                break

    return partition


def do_metis(G: nx.Graph, PG: nx.Graph) -> list[int]:
    (edgecuts, partition) = metis.part_graph(G, len(PG))
    return partition


def do_greed(G: nx.Graph, PG: nx.Graph, partition: list[int]) -> list[int]:
    print('initial partition: ', partition)
    print('f for initial partition: ', f(G, PG, partition))
    print('cut_ratio for initial partition: ', calc_cut_ratio(G, partition))
    partition = postprocessing_phase(G, PG, partition)
    print('final partition: ', partition)
    print('f for final partition: ', f(G, PG, partition))
    print('cut_ratio for final partition: ', calc_cut_ratio(G, partition))

    # color = []
    # for i, p in enumerate(partition):
    #     G.nodes[i]['color'] = 1/(p + 0.1)
    #     color.append(1/(p + 1))

    # nx.draw(G, node_color = color, with_labels=True, pos=nx.shell_layout(G))
    # plt.show()

    return partition


def write_results(path: str, physical_graph_path: str, partition: dict[int: list[int]], G: nx.Graph, PG: nx.Graph) -> None:
    global R2, ITER_MAX, PENALTY, CUT_RATIO

    HEADERS: list[str] = [
        'graph',
        'physical_graph',
        'cut_ratio',
        'PENALTY',
        'cut_ratio_limitation',
        'f',
        'partition',
    ]

    line2write = [
        path.split('/')[-1],
        physical_graph_path.split('/')[-1],
        PENALTY,
        calc_cut_ratio(G=G, partition=partition),
        CUT_RATIO,
        f(G, PG, partition),
        partition,
        '\n',
    ]

    with open(path, 'a+') as file:
        file.write(' '.join(map(str, line2write)))


def research() -> None:
    global CUT_RATIO
    graph_dirs = [
        (r'../data/sausages', r'../results/greed/{}sausages'),
        (r'../data/triangle/graphs', r'../results/greed/{}triangle'),
    ]

    physical_graph_dirs = [
        r'../data/physical_graphs',
    ]   

    print('3')


    for _ in range(5):
        for cr in [0.5, 0.6, 0.7, 0.8]:
            CUT_RATIO = cr
            for input_dir, output_dir in graph_dirs:
                for graph_file in listdir(input_dir):
                    # print(join(input_dir, graph_file))
                    if isfile(join(input_dir, graph_file)):
                        # print(join(input_dir, graph_file))
                        for physical_graph_dir in physical_graph_dirs:
                            for physical_graph_path in listdir(physical_graph_dir):
                                if isfile(join(physical_graph_dir, physical_graph_path)):
                                    weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
                                    unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))
                                    physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
                                    # weighted
                                    initial_weighted_partition = do_metis(weighted_graph, physical_graph)
                                    weighted_partition = do_greed(weighted_graph, physical_graph, initial_weighted_partition)
                                    write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                    # unweighted
                                    initial_unweighted_partition = do_metis(unweighted_graph, physical_graph)
                                    unweighted_partition = do_greed(weighted_graph, physical_graph, initial_unweighted_partition)
                                    write_results(join(output_dir.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)


# %%
if __name__ == '__main__':    
    # G = input_networkx_graph_from_file(r'../data/triangle/graphs/triadag10_0.txt')
    # G = input_networkx_graph_from_file(r'../data/sausages/dagA0.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/graph100.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/graph10.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/test1.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/gap2.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/hetero0.txt')
    # G = input_networkx_graph_from_file(r'../data/trash/hetero1.txt')

    # PG = input_networkx_graph_from_file(r'../data/test_gp/0.txt')
    # PG = input_networkx_graph_from_file(r'../data/test_gp/homo3.txt')

    # nx.draw(G, with_labels=True)
    # plt.show()


    # (edgecuts, partition) = metis.part_graph(G, len(PG))
    # colors = ['red','blue','green']
    # for i, p in enumerate(partition):
        # G.nodes[i]['color'] = colors[p]
    # nx.draw(G, with_labels=True)
    # plt.show()
    # print(edgecuts)
    research()
    

# %%
