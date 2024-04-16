from os import listdir, makedirs
from os.path import isfile, join, exists

from pathlib import Path

import sys
from time import sleep

import networkx as nx
import metis
from copy import deepcopy 


CUT_RATIO = 0.7
PENALTY = True
BIG_NUMBER = 1e10
MK_DIR = '../data_mk'


def input_networkx_graph_from_file(path: str) -> nx.Graph:
    G = nx.Graph()

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)

        return None

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

    if not isfile(path):
        if 'data_mk' in path:
            while not isfile(path):
                print('WAITING FOR MK GRAPH: ', path)
                sleep(10)
        return None

    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, _, *children = map(int, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            G.add_node(name, weight=1)
            G.add_edges_from((name, child) for child in children)
    return G


def f(G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None) -> float:
    p_loads = [0] * len(PG)

    if G is None or partition is None:
        return 2 * BIG_NUMBER

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


def calc_cut_ratio(G: nx.Graph | None, partition: list[int] | None) -> float | None:
    if G is None or partition is None:
        return None
    
    if len(G.edges) == 0:
        return 0
    
    # print('calc_edgecut(G, partition): ', calc_edgecut(G, partition))
    # print('len(G.edges): ', len(G.edges))

    return calc_edgecut(G, partition) / len(G.edges)


def check_cut_ratio(G: nx.Graph | None, partition: list[int] | None) -> bool:
    if G is None or partition is None:
        return False
    
    return calc_cut_ratio(G, partition) <= CUT_RATIO


def postprocessing_phase(G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None) -> list[int] | None:
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
    if partition is None:
        return None
    
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


def do_metis(G: nx.Graph | None, PG: nx.Graph, recursive: bool | None = None) -> list[int] | None:
    if recursive is None:
            if (len(PG) > 8):
                recursive = True
            else:
                recursive = False
    ufactor = 1
    # print(G)
    # print(G.nodes)
    # print(G.edges)

    if G is None:
        return None

    (edgecuts, partition2parce) = metis.part_graph(G, len(PG), objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)

    partition = [0] * len(G.nodes)
    
    # print(G.nodes)

    for new_i, i in enumerate(list(G.nodes)):
        partition[i] = partition2parce[new_i]

    assert edgecuts == calc_edgecut(G, partition)

    # print('edgecuts: ', edgecuts, 'ufactor', ufactor)
    # print('calc_edgecut: ', calc_edgecut(G, partition))
    # print('cut_ratio: ', calc_cut_ratio(G, partition))
    # # print(G)
    # print(partition)
    # print(*enumerate(partition))

    # sums = defaultdict(int)
    # for job, proc in enumerate(partition):
    #     sums[proc] += G.nodes[job]['weight']

    # print(sums)

    # raise Exception
    while not check_cut_ratio(G, partition):
        # print(ufactor)
        # print('cutratio: ', calc_cut_ratio(G, partition))
        ufactor *= 2

        if ufactor > 10e7:
            return None

        (edgecuts, partition2parce) = metis.part_graph(G, len(PG), objtype='cut', ncuts=10, ufactor=ufactor,)

        # partition = [0] * len(G.nodes)
        for new_i, i in enumerate(list(G.nodes)):
            partition[i] = partition2parce[new_i]

        # print(calc_edgecut(G, partition), set(partition))
        # print(partition.count(0), partition.count(1), partition.count(2), partition.count(3))

        assert edgecuts == calc_edgecut(G, partition)

        # print(edgecuts)
    
    ans = partition.copy()
    for _ in range(5):
        ufactor *= 0.75
        ufactor = int(ufactor)
        if ufactor < 1:
            break
        
        # print(ufactor)
        (edgecuts, partition) = metis.part_graph(G, len(PG), objtype='cut', ncuts=10, ufactor=ufactor,)
        if check_cut_ratio(G, partition):
            # print(G)
            # print(PG)
            if f(G, PG, partition) < f(G, PG, ans):
                ans = partition

    return ans


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

    # HEADERS: list[str] = [
    #     'graph',
    #     'physical_graph',
    #     'cut_ratio',
    #     'PENALTY',
    #     'cut_ratio_limitation',
    #     'f',
    #     'partition',
    # ]

    line2write = [
        path.split('/')[-1],
        physical_graph_path.split('/')[-1],
        PENALTY,
        calc_cut_ratio(G=G, partition=partition),
        CUT_RATIO,
        f(G, PG, partition),
        partition if check_cut_ratio(G, partition) else None,
        '\n',
    ]

    print(path, line2write)
    makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

    with open(path, 'a+') as file:
        file.write(' '.join(map(str, line2write)))


def unpack_mk(mk_partition: list[int], mk_data: list[int]) -> list[int]:
    ans = mk_data.copy()
    mapping = dict()

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


def research() -> None:
    global CUT_RATIO
    graph_dirs = [
        (r'../data/testing_graphs', r'../results/greed/{}testing_graphs'),
        (r'../data/triangle/graphs', r'../results/greed/{}triangle'),
        (r'../data/sausages', r'../results/greed/{}sausages'),
    ]

    physical_graph_dirs = [
        r'../data/physical_graphs',
    ]   

    print('3')

    cr_list = [0.6, 1]

    cr_list = [0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    cr_list = [0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    cr_list = [0.4]


    for input_dir, output_dir in graph_dirs:
        for graph_file in listdir(input_dir):
            # print(join(input_dir, graph_f ile))
            if isfile(join(input_dir, graph_file)) and 'dagD39.txt' in graph_file:
                # print(join(input_dir, graph_file))
                for physical_graph_dir in physical_graph_dirs:
                    for physical_graph_path in listdir(physical_graph_dir):
                        if isfile(join(physical_graph_dir, physical_graph_path)) and '3_2' in physical_graph_path:
                            for cr in cr_list:
                                # print(join(input_dir, graph_file))
                                CUT_RATIO = cr
                                weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
                                unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))
                                physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
                                # # weighted
                                initial_weighted_partition = do_metis(weighted_graph, physical_graph)
                                write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                weighted_partition = do_greed(weighted_graph, physical_graph, initial_weighted_partition)
                                write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                # # unweighted
                                initial_unweighted_partition = do_metis(unweighted_graph, physical_graph)
                                write_results(join(output_dir.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                                unweighted_partition = do_greed(weighted_graph, physical_graph, initial_unweighted_partition)
                                write_results(join(output_dir.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)
                                
                                print('from mk weighted')
                                CUT_RATIO = 1
                                output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')
                                mk_path = MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                                mk_data_path = MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')
                                mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                                mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                                physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                                # print(join(input_dir, graph_file))
                                # print(mk_path)

                                initial_weighted_partition = do_metis(mk_graph_weighted, physical_graph)
                                initial_unweighted_partition = do_metis(mk_graph_unweighted, physical_graph)
                                weighted_partition = do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                                unweighted_partition = do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                                initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                                initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                                weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                                unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                                # weighted_partition = do_greed(weighted_graph, physical_graph, initial_weighted_partition)
                                # unweighted_partition = do_greed(unweighted_graph, physical_graph, initial_unweighted_partition)

                                CUT_RATIO = cr
                                # write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                # write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                print('unweighted')
                                write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)


                                print('from mk UNweighted')
                                CUT_RATIO = 1
                                output_dir_mk = output_dir.replace('greed', 'greed_from_mk_unweighted')
                                mk_path = MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.', str(cr) + '.')
                                mk_data_path = MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.txt', str(cr) + '.' + 'mapping')
                                mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                                mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                                physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                                # print(join(input_dir, graph_file))
                                # print(mk_path)

                                initial_weighted_partition = do_metis(mk_graph_weighted, physical_graph)
                                initial_unweighted_partition = do_metis(mk_graph_unweighted, physical_graph)
                                weighted_partition = do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                                unweighted_partition = do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                                initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                                initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                                weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                                unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                                CUT_RATIO = cr
                                # write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                # write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                # unweighted
                                write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                                write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)


def run() -> None:
    ...


if __name__ == '__main__':
    match sys.argv[1:]:
        case ['--G', g_path, '--PG', pg_path, '--CR', cr, '--OUTPUT_DIR', output_dir]:
            CUT_RATIO = cr

            weighted_graph = input_networkx_graph_from_file(g_path)
            unweighted_graph = input_networkx_unweighted_graph_from_file(g_path)
            physical_graph = input_networkx_graph_from_file(pg_path)

            if output_dir.endswith('{}'):
                pass
            elif output_dir.endswith('/'):
                output_dir += {}
            else:
                output_dir += '/{}'

            graph_name = g_path.split('/')[-1]

            # # weighted
            # initial_weighted_partition = do_metis(weighted_graph, physical_graph)
            # write_results(join(output_dir.format('weighted/'), graph_name).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
            # weighted_partition = do_greed(weighted_graph, physical_graph, initial_weighted_partition)
            # write_results(join(output_dir.format('weighted/'), graph_name), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

            # # unweighted
            # initial_unweighted_partition = do_metis(unweighted_graph, physical_graph)
            # write_results(join(output_dir.format('unweighted/'), graph_name).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
            # unweighted_partition = do_greed(weighted_graph, physical_graph, initial_unweighted_partition)
            # write_results(join(output_dir.format('unweighted/'), graph_name), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)
        case _:
            research()
