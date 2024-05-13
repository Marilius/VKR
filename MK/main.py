from helpers import input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file, calc_edgecut, calc_cut_ratio
from base_partitioner import BasePartitioner

import networkx as nx
import metis

from os import listdir, makedirs
from os.path import isfile, join

import os


class MK(BasePartitioner):
    def __init__(self, data_dirs: list[str]) -> None:
        self.CUT_RATIO = 0.3
        self.BIG_NUMBER = 1e10
        self.MAX_UFACTOR = 1e4
        self.output_folder = './data_mk/{}'
        self.CACHE_DIR: str = './mk_cache'

        self.data_dirs = data_dirs

    def write_mk(self, g_name: str, G_grouped: nx.Graph, mk_partition: list[int]) -> None:
        if self.CUT_RATIO == 1:
            self.CUT_RATIO = 1

        output_file = self.output_folder.format(g_name.replace('.', str(self.CUT_RATIO) + '.'))

        # print(output_file, G_grouped)
        # print(sorted(list(G_grouped.nodes)))

        with open(output_file, 'w+') as file:
            # if os.path.getsize(self.output_folder.format(g_name)) == 0:
            file.write('name weight children\n')

            for node_id in sorted(list(G_grouped.nodes)):
                line = [str(node_id), str(G_grouped.nodes[node_id]['weight'])]

                for neighbor in G_grouped.neighbors(node_id):
                    if neighbor > node_id:
                        line.append(str(neighbor)) 

                line.append('\n')
                # print(f"'{line}'")
                file.write(' '.join(line))

        output_file = self.output_folder.format(g_name.replace('.txt', str(self.CUT_RATIO) + '.' + 'mapping'))
        with open(output_file, 'w+') as file:
            file.write(' '.join(map(str, mk_partition)))
        # print(self.CUT_RATIO, len(set(mk_partition)), [mk_partition.count(i) for i in sorted(list(set(mk_partition)))])
        # sleep(7)

    def do_metis(self, G: nx.Graph, nparts: int, ufactor: int, recursive: bool | None = None) -> list[int]:
        if recursive is None:
            if (nparts > 8):
                recursive = True
            else:
                recursive = False

        if nparts == 1:  # Floating point exception from metis ¯\_(ツ)_/¯
            return [0] * len(G.nodes)

        (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)

        partition = [0] * len(G.nodes)

        for new_i, i in enumerate(list(G.nodes)):
            partition[i] = partition2parse[new_i]

        # print('++++>', partition)

        for new_i, i in enumerate(sorted(list(set(partition)))):
            for j in range(len(partition)):
                if partition[j] == i:
                    partition[j] = new_i

        # print('++++>', partition)

        assert edgecuts == calc_edgecut(G, partition)
        assert len(partition) == len(G.nodes)

        return partition
    
    def load_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool) -> list[int] | None:
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(nparts) + '!_' + str(cr) + '_' + str(weighted) + '.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, partition: list[int] | None) -> None:        
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(nparts) + '!_' + str(cr) + '_' + str(weighted) + '.txt'
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def mk_nparts(self, G: nx.Graph, nparts: int, cr: float | None = None, max_ufactor: float | None = 1e4, weighted: bool = True, check_cache: bool = True) -> tuple[nx.Graph, list[int]] | tuple[None, None]:
        if cr is not None:
            self.CUT_RATIO = cr

        if max_ufactor is not None:
            self.MAX_UFACTOR = max_ufactor

        if check_cache:
            partition = self.load_mk_nparts_cache(G, nparts, self.CUT_RATIO, weighted)
            if partition is not None:
                G_grouped = self.group_mk(G, partition, weighted=weighted)
                G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(self.CUT_RATIO) + '_' + str(weighted)
                return (G_grouped, partition)

        n_ans = 0
        partition_ans = None

        ufactor = 1
        while ufactor < self.MAX_UFACTOR: 
            partition = self.do_metis(G, nparts, ufactor)

            if self.check_cut_ratio(G, partition):
                if len(set(partition)) == nparts:
                    n_ans = nparts
                    partition_ans = partition
                    break

                if len(set(partition)) > n_ans:
                    n_ans = len(set(partition))
                    partition_ans = partition

            # ufactor += min(100, ufactor)
            ufactor += ufactor
        
        if partition_ans is None:
            return (None, None)
        
        G_grouped = self.group_mk(G, partition_ans, weighted=weighted)

        print('--->', G_grouped.nodes)
        print('--->', partition_ans)

        if check_cache:
            self.write_mk_nparts_cache(G, nparts, self.CUT_RATIO, weighted, partition_ans)

        G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(self.CUT_RATIO) + '_' + str(weighted)

        return (G_grouped, partition_ans)

    def mk_part(self, G: nx.Graph) -> tuple[int, list[int]]:
        num_left = 2
        num_right = len(G)

        n_ans = 0
        partition_ans = None

        n = 0
        ufactor = 0

        while num_right - num_left > 1:
            # print('--------')
            # print(num_left, num_right)
            n = (num_left + num_right) // 2
            print(n)

            ufactor = 1
            while True: 
                partition = self.do_metis(G, n, ufactor)
                # print(partition)
                # print(calc_cut_ratio(G, partition))

                # print(self.check_cut_ratio(G, partition))

                if self.check_cut_ratio(G, partition):
                    print('was here')
                    if len(set(partition)) == n:
                        num_left = n + 1
                        n_ans = n
                        partition_ans = partition
                    else:
                        num_right = n

                    if len(set(partition)) > n_ans:
                        num_left = len(set(partition)) - 1
                        n_ans = len(set(partition))
                        partition_ans = partition

                    break

                if ufactor > self.MAX_UFACTOR:
                    print('ENDED BY UFACTOR')
                    num_right = n - 1
                    break

                # TODO как менять ufactor?
                ufactor += ufactor
            # print('--------')

        print('main ended')

        partition = self.do_metis(G, num_right, ufactor)
        # print(partition)
        # print(calc_cut_ratio(G, partition))

        if self.check_cut_ratio(G, partition):
            if len(set(partition)) > n_ans:
                n_ans = num_right
                partition_ans = partition

        partition = self.do_metis(G, num_left, ufactor)
        # print(partition)
        # print(calc_cut_ratio(G, partition))

        if self.check_cut_ratio(G, partition):
            if len(set(partition)) > n_ans:
                n_ans = num_left
                partition_ans = partition

        assert partition_ans, partition_ans

        if set(range(n_ans)) != set(partition_ans):
            mapping = dict()

            # print(partition_ans)

            for new_id, old_id in enumerate(set(partition_ans)):
                mapping[old_id] = new_id

            for i in range(len(partition_ans)):
                partition_ans[i] = mapping[partition_ans[i]]

            # print(partition_ans)

            # raise Exception

        return (n_ans, partition_ans)

    def get_num_mk(self, G: nx.Graph, cr: float | None = None) -> int:
        if cr is not None:
            self.CUT_RATIO = cr

        (n, _) = self.mk_part(G)

        return n

    def group_mk(self, G: nx.Graph, partition: list[int], weighted: bool = True) -> nx.Graph:
        grouped_G = nx.Graph()
        nodes_ids = sorted(list(set(partition)))

        for node_id in nodes_ids:
            weight = 0
            for num, part in enumerate(partition):
                if part == node_id:
                    weight += G.nodes[num]['weight']
            grouped_G.add_node(node_id, weight=weight)

        for old_node_id, node_id in enumerate(partition):
            for old_neighbor_id in G.neighbors(old_node_id):
                neighbor_id = partition[old_neighbor_id]
                if node_id != neighbor_id:
                    grouped_G.add_edge(node_id, neighbor_id)

        if weighted:
            grouped_G.graph['node_weight_attr'] = 'weight'

        # grouped_G.graph['graph_name'] = G.graph['graph_name'] + '_grouped'

        return grouped_G

    def research(self) -> None:
        # cr_list_little = [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
        # cr_list_big = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        # cr_list = cr_list_little + cr_list_big

        cr_list = [i/100 for i in range(5, 100)] + [1]
        cr_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        for input_dir in self.data_dirs:
            for graph_file in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, graph_file)) and 'dagP38' in graph_file:
                    g_path = os.path.join(input_dir, graph_file)
                    print(g_path)
                    for cr in cr_list:
                        print(cr)
                        print('!!!!!!!', g_path, '!!!!!!!!!!!')

                        self.CUT_RATIO = cr

                        # weighted
                        G_weighted = input_networkx_graph_from_file(g_path)

                        (_, weighted_partition) = self.mk_part(G_weighted)

                        if weighted_partition is None:
                            print(g_path, self.CUT_RATIO)
                            continue

                        mk_graph = self.group_mk(G_weighted, weighted_partition)
                        self.write_mk(graph_file.replace('.', '_weighted.'), mk_graph, weighted_partition)

                        # unweighted
                        G_unweighted = input_networkx_unweighted_graph_from_file(g_path)

                        (_, unweighted_partition) = self.mk_part(G_unweighted)

                        if unweighted_partition is None:
                            print(g_path, self.CUT_RATIO)
                            continue

                        mk_graph = self.group_mk(G_unweighted, unweighted_partition)

                        self.write_mk(graph_file.replace('.', '_unweighted.'), mk_graph, unweighted_partition)
