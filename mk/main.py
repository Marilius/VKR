from helpers import input_graph, calc_cut_ratio, do_unpack_mk, unpack_mk

import base_partitioner.settings as settings

from greed.main import GreedPartitioner

import networkx as nx

from os import makedirs
from os.path import isfile, join

import time


class MKPartitioner(GreedPartitioner):
    def write_mk(self, g_name: str, G_grouped: nx.Graph, mk_partition: list[int], cr_max: float) -> None:
        if cr_max == 1:
            cr_max = 1

        output_file = settings.MK_DIR + g_name.replace('.', str(cr_max) + '.')

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
        output_file = settings.MK_DIR + g_name.replace(ending2replace, str(cr_max) + '.' + 'mapping')
        with open(output_file, 'w+') as file:
            file.write(' '.join(map(str, mk_partition)))

    def load_mk_nparts_cache(self, G: nx.Graph, nparts: int, cr: float, weighted: bool, steps_back: int) -> list[int] | None:
        w = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = settings.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(steps_back) + '!_' + str(cr) + '_' + str(weighted) + '.txt'

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
        path = settings.CACHE_DIR + '/' + G.graph['graph_name'] + '_!' + str(nparts) + '!_' + w + str(steps_back) + '!_' + str(cr) + '_' + str(weighted) + '.txt'
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def mk_nparts(self, G: nx.Graph, nparts: int, cr_max: float, check_cache: bool, seed: int | None, max_ufactor: float | None = 1e7, weighted: bool = True, steps_back: int = 5) -> tuple[nx.Graph, list[int]] | tuple[None, None]:
        if max_ufactor is not None:
            self.MAX_UFACTOR = max_ufactor

        # if check_cache:
        #     partition = self.load_mk_nparts_cache(G, nparts, cr_max, weighted, steps_back=steps_back)
        #     if partition is not None:
        #         G_grouped = self.group_mk(G, partition, weighted=weighted)
        #         G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(cr_max) + '_' + str(weighted)
        #         return (G_grouped, partition)
        # print('dddddd', len(G.nodes))
        partition_ans = super().do_metis(G, nparts, cr_max, check_cache, seed, steps_back=steps_back)
        # print(G.graph['graph_name'])
        # print('dddddd', len(G.nodes), len(partition_ans))
        print('cccccccccccc', partition_ans)

        if partition_ans is None:
            return (None, None)

        G_grouped = self.group_mk(G, partition_ans, weighted=weighted)

        print('--->', G_grouped.nodes)
        print('--->', partition_ans)

        # if check_cache:
        #     self.write_mk_nparts_cache(G, nparts, cr_max, weighted, partition_ans, steps_back=steps_back)

        G_grouped.graph['graph_name'] = G.graph['graph_name'] + '_grouped_' + str(nparts) + '_' + str(cr_max) + '_' + str(weighted)

        return (G_grouped, partition_ans)

    def load_mk_part_cache(self, G: nx.Graph, cr_max: float, steps_back: int) -> list[int] | None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/mk_part/{G_hash}_{cr_max}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                partition = list(map(int, line.split()))
                return partition

        return None

    def write_mk_part_cache(self, G: nx.Graph, partition: list[int], cr_max: float, steps_back: int) -> None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/mk_part/{G_hash}_{cr_max}_{steps_back}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            file.write(' '.join(map(str, partition)))

    def mk_part(self, G: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 5) -> tuple[int, list[int]]:
        if check_cache:
            partition = self.load_mk_part_cache(G, cr_max, steps_back=steps_back)
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
            print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, settings.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

            ufactor = 1
            while True: 
                (_, partition) = self.metis_part(G, n, ufactor, check_cache, seed)
                print(n, len(set(partition)), ufactor, calc_cut_ratio(G, partition))
                if self.check_cut_ratio(G, partition, cr_max):
                    print('was here')
                    partition_curr = partition.copy()

                    for _ in range(steps_back):
                        ufactor *= 0.75
                        ufactor = int(ufactor)
                        if ufactor < 1:
                            break

                        (_, partition) = self.metis_part(G, n, ufactor, check_cache, seed)
                        if len(set(partition_curr)) < len(set(partition)):
                            partition_curr = partition.copy()

                    if len(set(partition_curr)) > n_ans:
                        num_left = len(set(partition_curr)) + 1
                        n_ans = len(set(partition_curr))
                        partition_ans = partition_curr
                    else:
                        num_right = n

                    break

                if ufactor > settings.MAX_UFACTOR:
                    print('ENDED BY UFACTOR')
                    num_right = n - 1
                    break

                ufactor += ufactor

        print('main ended')

        ufactor = 1
        while ufactor < settings.MAX_UFACTOR:
            (_, partition) = self.metis_part(G, num_right, ufactor, check_cache, seed)
            if self.check_cut_ratio(G, partition, cr_max):
                if len(set(partition)) > n_ans:
                    n_ans = len(set(partition))
                    partition_ans = partition
                break
            ufactor *= 2

        print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, settings.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

        if set(range(n_ans)) != set(partition_ans):
            mapping = dict()

            for new_id, old_id in enumerate(set(partition_ans)):
                mapping[old_id] = new_id

            for i in range(len(partition_ans)):
                partition_ans[i] = mapping[partition_ans[i]]

        print('n_ans, n, left, right= ', n_ans, n, num_left, num_right, settings.MAX_UFACTOR, calc_cut_ratio(G, partition_ans))

        if check_cache:
            self.write_mk_part_cache(G, partition_ans, cr_max, steps_back=steps_back)

        return (n_ans, partition_ans)

    def get_num_mk(self, G: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 5, ) -> int:
        (n, _) = self.mk_part(G, cr_max, check_cache, seed, steps_back=steps_back)

        return n

    def group_mk(self, G: nx.Graph, partition: list[int], weighted: bool = True) -> nx.Graph:
        grouped_G = nx.Graph()
        nodes_ids = sorted(list(set(partition)))

        print(sorted(G.nodes))
        print(len(G.nodes), len(partition))
        print(G.nodes(data=True))
        print(partition)

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
            
            # if node_id not in grouped_G.nodes or 'weight' not in grouped_G.nodes(data=True)[node_id]:
            #     grouped_G.add_node(node_id)
            #     grouped_G.nodes[node_id]['weight'] = G.nodes[old_node_id]['weight']
            # else:
            #     # print(grouped_G.nodes(data=True)[node_id])
            #     grouped_G.nodes[node_id]['weight'] += G.nodes[old_node_id]['weight']
                    
        # TODO ^ error

        if weighted:
            grouped_G.graph['node_weight_attr'] = 'weight'

        grouped_G.graph['graph_name'] = G.graph['graph_name'] + '_grouped'

        return grouped_G


    def MK_greed_greed(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 6, ) -> list[int] | None:
        max_mk = self.get_num_mk(G, cr_max, check_cache, seed, steps_back=steps_back)

        best_partition: list[int] = [0] * len(G.nodes)
        best_f: float = self.f(G, PG, best_partition, cr_max)

        n = 0
        if self.do_metis_with_pg(G, PG, cr_max, check_cache, seed):
            part = self.do_metis_with_pg(G, PG, cr_max, check_cache, seed)
            if part is not None:
                n = len(set(part))
            else:
                n = 1

        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], cr_max)

        for nparts in range(1, max_mk + 1):
            (G_grouped, mk_data) = self.mk_nparts(G, nparts, cr_max, check_cache, seed, steps_back=steps_back)
            print('bbbbbbbbbbb', G_grouped, mk_data)

            if G_grouped is None or mk_data is None:
                continue

            mk_partition = self.do_metis_with_pg(G_grouped, PG, 1, check_cache, seed, steps_back=steps_back)
            # print('aaaaaaaaa', mk_partition)
            # print('aaaaaaaaa', G_grouped)
            mk_partition = self.do_greed(G_grouped, PG, mk_partition, 1)

            if mk_partition is None:
                continue
            
            try:
                mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
            except Exception as e:
                print(mk_partition, mk_data)
                raise e
            if self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                partition = self.do_greed(G, PG, mk_partition_unpacked, cr_max)
                f_val = self.f(G, PG, partition, cr_max)

                if f_val < best_f:
                    best_f = f_val
                    best_partition = partition.copy()

        return best_partition
    
    def do_MK_greed_greed(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        steps_back: int,
        check_cache: bool,
        seed: int | None,
    ) -> list[int] | None:
        output_dir = output_dir.replace('results', 'results2')
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_weighted')

        start_time = time.time()
        partition = self.MK_greed_greed(weighted_graph, physical_graph, cr_max, check_cache, seed, steps_back=steps_back)

        # assert self.f(weighted_graph, physical_graph, partition) <= self.f(weighted_graph, physical_graph, self.just_weighted_partition)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, cr_max, start_time)

        return partition

    def MK_greed_greed_with_geq_cr(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, steps_back: int = 6) -> list[int]:
        best_best_partition: list[int] = [0] * len(G.nodes)
        best_best_f: float = self.f(G, PG, best_best_partition, cr_max)

        max_mk = self.get_num_mk(G, cr_max, check_cache, seed, steps_back=steps_back)

        n = 0
        if self.do_metis_with_pg(G, PG, cr_max):
            n = len(set(self.do_metis_with_pg(G, PG, cr_max)))
        
        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], cr_max)

        for cr in settings.ALL_CR_LIST:
            if cr >= cr_max:
                best_partition: list[int] = [0] * len(G.nodes)
                best_f: float = self.f(G, PG, best_partition, cr)

                f: bool = False

                for nparts in range(1, max_mk + 1):
                    (G_grouped, mk_data) = self.mk_nparts(G, nparts, cr, check_cache=check_cache, steps_back=steps_back)

                    if G_grouped is None or mk_data is None:
                        continue

                    mk_partition = self.do_metis_with_pg(G_grouped, PG, 1, check_cache=check_cache, steps_back=steps_back)
                    mk_partition = self.do_greed(G_grouped, PG, mk_partition, 1)

                    if mk_partition is None:    
                        continue

                    mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
                    if not self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                        f = True
                        break

                    if self.check_cut_ratio(G, mk_partition_unpacked, cr_max):
                        partition = self.do_greed(G, PG, mk_partition_unpacked, cr_max)
                        f_val = self.f(G, PG, partition, cr_max)
                        if f_val < best_f:
                            best_f = f_val
                            best_partition = partition.copy()

                if best_f < best_best_f:
                    best_best_f = best_f
                    best_best_partition = best_partition.copy()

                if f:
                    break

        if best_f < best_best_f:
            best_best_f = best_f
            best_best_partition = best_partition.copy() 

        return best_best_partition

    def do_MK_greed_greed_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        seed: int | None,
        check_cache: bool,
        steps_back: int = 6,
    ) -> None:
        output_dir = output_dir.replace('results', 'results2')
        
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_with_geq_cr')

        start_time = time.time()
        partition = self.MK_greed_greed_with_geq_cr(weighted_graph, physical_graph, cr_max, seed, check_cache=check_cache, steps_back=steps_back)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, cr_max, start_time)

    def do_weighted_mk_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
    ) -> None:
        output_dir = output_dir.replace('results', 'results1')
        weighted_graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')

        ans_init = None
        ans_part = None
        ans = None

        for cr in settings.ALL_CR_LIST:
            if cr >= cr_max:
                mk_path = settings.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                mk_data_path = settings.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')

                if not isfile(mk_path) or not isfile(mk_data_path):
                    continue

                mk_graph_weighted = input_graph(mk_path)

                initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, 1, check_cache=True)
                assert initial_weighted_partition is not None
                weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition, 1)
                assert weighted_partition is not None

                initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                assert initial_weighted_partition is not None
                weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                assert weighted_partition is not None

                if not self.check_cut_ratio(weighted_graph, weighted_partition, cr_max):
                    break

                if ans_part is None or ans is None or self.f(weighted_graph, physical_graph, weighted_partition, cr_max) < ans:
                    ans_init = initial_weighted_partition
                    ans_part = weighted_partition
                    ans = self.f(weighted_graph, physical_graph, weighted_partition, cr_max)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), ans_init, weighted_graph, physical_graph, cr_max)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), ans_part, weighted_graph, physical_graph, cr_max)
