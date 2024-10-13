from helpers import input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file, calc_cut_ratio, input_generated_graph_and_processors_from_file
from base_partitioner import BasePartitioner

import networkx as nx

from os import makedirs
from os.path import isfile

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

        # print(g_name)
        # print(g_name)

        output_file = self.output_folder.format(g_name.replace('.', str(self.CUT_RATIO) + '.'))
        # print(output_file)

        with open(output_file, 'w+') as file:
            file.write('name weight children\n')

            for node_id in sorted(list(G_grouped.nodes)):
                line = [str(node_id), str(G_grouped.nodes[node_id]['weight'])]

                for neighbor in G_grouped.neighbors(node_id):
                    if neighbor > node_id:
                        line.append(str(neighbor)) 

                line.append('\n')
                file.write(' '.join(line))

        ending2replace = '.txt' if 'txt' in g_name else '.graph'
        output_file = self.output_folder.format(g_name.replace(ending2replace, str(self.CUT_RATIO) + '.' + 'mapping'))
        with open(output_file, 'w+') as file:
            file.write(' '.join(map(str, mk_partition)))
    
    def load_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, steps_back: int) -> list[int] | None:
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(steps_back) + '!_' + str(cr) + '_' + str(weighted) + '.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, partition: list[int] | None, steps_back: int) -> None:        
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(steps_back) + '!_' + str(cr) + '_' + str(weighted) + '.txt'
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def mk_nparts(self, G: nx.Graph, nparts: int, cr: float | None = None, max_ufactor: float | None = 1e7, weighted: bool = True, check_cache: bool = True, steps_back: int = 5) -> tuple[nx.Graph, list[int]] | tuple[None, None]:
        if cr is not None:
            self.CUT_RATIO = cr

        if max_ufactor is not None:
            self.MAX_UFACTOR = max_ufactor

        # if check_cache:
        #     partition = self.load_mk_nparts_cache(G, nparts, self.CUT_RATIO, weighted, steps_back=steps_back)
        #     if partition is not None:
        #         G_grouped = self.group_mk(G, partition, weighted=weighted)
        #         G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(self.CUT_RATIO) + '_' + str(weighted)
        #         return (G_grouped, partition)

        partition_ans = super().do_metis(G, nparts, steps_back=steps_back)

        if partition_ans is None:
            return (None, None)

        G_grouped = self.group_mk(G, partition_ans, weighted=weighted)

        print('--->', G_grouped.nodes)
        print('--->', partition_ans)

        # if check_cache:
        #     self.write_mk_nparts_cache(G, nparts, self.CUT_RATIO, weighted, partition_ans, steps_back=steps_back)

        G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(self.CUT_RATIO) + '_' + str(weighted)

        return (G_grouped, partition_ans)
    
    def load_mk_part_cache(self, G: nx.Graph, steps_back: int) -> list[int] | None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{self.CACHE_DIR}/mk_part/{G_hash}_{self.CUT_RATIO}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                partition = list(map(int, line.split()))
                return partition
        
        return None

    def write_mk_part_cache(self, G: nx.Graph, partition: list[int], steps_back: int) -> None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{self.CACHE_DIR}/mk_part/{G_hash}_{self.CUT_RATIO}_{steps_back}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            file.write(' '.join(map(str, partition)))

    def mk_part(self, G: nx.Graph, check_cache: bool = True, steps_back: int = 5) -> tuple[int, list[int]]:
        if check_cache:
            partition = self.load_mk_part_cache(G, steps_back=steps_back)
            if partition is not None:
                return (len(set(partition)), partition)

        num_left = 1
        num_right = len(G)

        n_ans = 0
        partition_ans = None

        n = 0
        ufactor = 0

        while num_right - num_left > 0:
            n = (num_left + num_right) // 2
            print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, self.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

            ufactor = 1
            while True: 
                (_, partition) = self.metis_part(G, n, ufactor)
                print(n, len(set(partition)), ufactor, calc_cut_ratio(G, partition))
                if self.check_cut_ratio(G, partition):
                    print('was here')
                    partition_curr = partition.copy()
                
                    for _ in range(steps_back):
                        ufactor *= 0.75
                        ufactor = int(ufactor)
                        if ufactor < 1:
                            break

                        (_, partition) = self.metis_part(G, n, ufactor)
                        if len(set(partition_curr)) < len(set(partition)):
                            partition_curr = partition.copy()
                            
                    if len(set(partition_curr)) > n_ans:
                        num_left = len(set(partition_curr)) + 1
                        n_ans = len(set(partition_curr))
                        partition_ans = partition_curr
                    else:
                        num_right = n

                    break

                if ufactor > self.MAX_UFACTOR:
                    print('ENDED BY UFACTOR')
                    num_right = n - 1
                    break

                ufactor += ufactor

        print('main ended')

        ufactor = 1
        while ufactor < self.MAX_UFACTOR:
            (_, partition) = self.metis_part(G, num_right, ufactor)
            if self.check_cut_ratio(G, partition):
                if len(set(partition)) > n_ans:
                    n_ans = len(set(partition))
                    partition_ans = partition
                break
            ufactor *= 2

        # ufactor = 1
        # while ufactor < self.MAX_UFACTOR:
        #     (_, partition) = self.metis_part(G, num_left, ufactor)
        #     if self.check_cut_ratio(G, partition):
        #         if len(set(partition)) > n_ans:
        #             n_ans = len(set(partition))
        #             partition_ans = partition
        #         break
        #     ufactor *= 2

        # assert partition_ans, partition_ans

        print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, self.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

        if set(range(n_ans)) != set(partition_ans):
            mapping = dict()

            for new_id, old_id in enumerate(set(partition_ans)):
                mapping[old_id] = new_id

            for i in range(len(partition_ans)):
                partition_ans[i] = mapping[partition_ans[i]]

        print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, self.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

        if check_cache:
            self.write_mk_part_cache(G, partition_ans, steps_back=steps_back)

        return (n_ans, partition_ans)

    def get_num_mk(self, G: nx.Graph, cr: float | None = None, steps_back: int = 5, check_cache: bool = True) -> int:
        if cr is not None:
            self.CUT_RATIO = cr

        (n, _) = self.mk_part(G, steps_back=steps_back, check_cache=check_cache)

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

        grouped_G.graph['graph_name'] = G.graph['graph_name'] + '_grouped'

        return grouped_G

    def research(self) -> None:
        cr_list = [i/100 for i in range(5, 100)] + [1]
        # cr_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        for input_dir in self.data_dirs:
            for graph_file in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, graph_file)) and 'partition' not in graph_file:
                    g_path = os.path.join(input_dir, graph_file)
                    print(g_path)
                    for cr in cr_list:
                        print(cr)
                        print('!!!!!!!', g_path, '!!!!!!!!!!!')

                        self.CUT_RATIO = cr

                        G_weighted: nx.Graph
                        # weighted
                        if 'gen_data' in g_path:
                            G_weighted, _, _ = input_generated_graph_and_processors_from_file(g_path)
                        else:
                            G_weighted = input_networkx_graph_from_file(g_path)

                        print(G_weighted)
                        (_, weighted_partition) = self.mk_part(G_weighted)

                        if weighted_partition is None:
                            print(g_path, self.CUT_RATIO)
                            continue

                        print(G_weighted.graph['graph_name'])
                        mk_graph = self.group_mk(G_weighted, weighted_partition)
                        print(mk_graph.graph['graph_name'])
                        self.write_mk(graph_file + '_weighted', mk_graph, weighted_partition)
                        # self.write_mk(graph_file.replace('.', '_weighted.'), mk_graph, weighted_partition)

                        # # unweighted
                        # G_unweighted = input_networkx_unweighted_graph_from_file(g_path)

                        # (_, unweighted_partition) = self.mk_part(G_unweighted)

                        # if unweighted_partition is None:
                        #     print(g_path, self.CUT_RATIO)
                        #     continue

                        # mk_graph = self.group_mk(G_unweighted, unweighted_partition)

                        # self.write_mk(graph_file.replace('.', '_unweighted.'), mk_graph, unweighted_partition)

    def part_graph(self, graph: nx.Graph):
        cr_list = [i/100 for i in range(5, 100)] + [1]

        for cr in cr_list:
            self.CUT_RATIO = cr

            # weighted

            (_, partition) = self.mk_part(graph)

            if partition is None:
                continue

            mk_graph = self.group_mk(graph, partition)
            self.write_mk(graph_file.replace('.', '_weighted.'), mk_graph, partition)
