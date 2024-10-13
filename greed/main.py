from helpers import input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file, calc_cut_ratio, do_unpack_mk, unpack_mk, input_generated_graph_and_processors_from_file
from MK import MK
from base_partitioner import BasePartitioner

from os import listdir, makedirs
from os.path import isfile, join

import networkx as nx

import time


class Greed(BasePartitioner):
    def __init__(self) -> None:
        self.MK_DIR: str = './data_mk'
        self.CACHE_DIR: str = './cache'
        self.ALL_CR_LIST: list[float] = [i/100 for i in range(4, 100)] + [1]
        self.BIG_NUMBER: float = 1e10
        self.PENALTY: bool = True
        self.CUT_RATIO: float = 0.7
        self.mk: MK = MK([])

    def write_results(self, path: str, physical_graph_path: str, partition: list[int], G: nx.Graph, PG: nx.Graph, start_time: float) -> None:
        # HEADERS: list[str] = [
        #     'graph',
        #     'physical_graph',
        #     'cut_ratio',
        #     'PENALTY',
        #     'cut_ratio_limitation',
        #     'f',
        #     'partition',
        # ]

        end_time = time.time()

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1],
            self.PENALTY,
            calc_cut_ratio(G=G, partition=partition),
            self.CUT_RATIO,
            self.f(G, PG, partition),
            partition if self.check_cut_ratio(G, partition) else None,
            '\n',
        ]

        # assert partition is None or len(G) == len(partition), (path, physical_graph_path, self.CUT_RATIO)
        assert partition is None or len(set(partition)) <= len(PG.nodes)

        print(path, line2write)
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

        overall_time = end_time - start_time

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1],
            self.CUT_RATIO,
            len(partition),
            start_time,
            str(overall_time),
            '\n',
        ]

        with open(path.replace('.txt', '.time'), 'a+') as file:
            file.write(' '.join(map(str, line2write)))

    def postprocessing_phase(self, G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None) -> list[int] | None:
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
        if partition is None or G is None:
            return None

        p_loads = [0] * len(PG)
        p_order: list[int] = list(range(len(PG)))
        p_order.sort(key=lambda i: PG.nodes[i]['weight'], reverse=True)

        for i in range(len(partition)):
            p_loads[partition[i]] += G.nodes[i]['weight']

        flag = True
        while flag:
            if 'mk_' in G.graph['graph_name']:
                print(partition)

            p1 = None
            p1_time = 0
            for i, i_load in enumerate(p_loads):
                if i_load / PG.nodes[i]['weight'] > p1_time or p1 is None:
                    p1 = i
                    p1_time = i_load / PG.nodes[i]['weight']

            if p1 is None:
                break

            while flag:
                flag = False

                a = None
                a_weight = 0
                for job, proc in enumerate(partition):
                    if proc == p1:
                        if G.nodes[job]['weight'] > a_weight:
                            a = job
                            a_weight = G.nodes[job]['weight']

                if a is None:
                    break

                for proc in p_order:
                    if proc != p1:
                        if max(p_loads[proc] / PG.nodes[proc]['weight'], p_loads[p1] / PG.nodes[p1]['weight']) \
                                > max((p_loads[p1] - a_weight)/ PG.nodes[p1]['weight'], (p_loads[proc] + a_weight) / PG.nodes[proc]['weight']):
                            partition_copy = partition.copy()
                            partition_copy[a] = proc
                            if self.check_cut_ratio(G, partition_copy):
                                p_loads[proc] += a_weight
                                p_loads[p1] -= a_weight
                                partition[a] = proc
                                flag = True
                                break

                if flag:
                    break

        return partition
    
    def MK_greed_greed(self, G: nx.Graph, PG: nx.Graph, steps_back: int = 6, check_cache: bool = True) -> list[int] | None:
        max_mk = self.mk.get_num_mk(G, self.CUT_RATIO, steps_back=steps_back, check_cache=check_cache)

        best_partition: list[int] = [0] * len(G.nodes)
        best_f: float = self.f(G, PG, best_partition)

        n = 0
        if self.do_metis_with_pg(G, PG):
            n = len(set(self.do_metis_with_pg(G, PG)))

        cr0 = self.CUT_RATIO

        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], self.CUT_RATIO)

        for nparts in range(1, max_mk + 1):
            (G_grouped, mk_data) = self.mk.mk_nparts(G, nparts, self.CUT_RATIO, check_cache=check_cache, steps_back=steps_back)

            if G_grouped is None or mk_data is None:
                continue

            self.CUT_RATIO = 1
            mk_partition = self.do_metis_with_pg(G_grouped, PG, check_cache=check_cache, steps_back=steps_back)
            mk_partition = self.do_greed(G_grouped, PG, mk_partition)
            self.CUT_RATIO = cr0

            if mk_partition is None:
                continue

            mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
            if self.check_cut_ratio(G, mk_partition_unpacked):
                print("check_passed")
                partition = self.do_greed(G, PG, mk_partition_unpacked)
                f_val = self.f(G, PG, partition)
                print('f_mk_partition_unpacked: ', self.f(G, PG, mk_partition_unpacked), 'f_val: ', f_val, calc_cut_ratio(G, mk_partition_unpacked))
                print(f_val, best_f)
                if f_val < best_f:
                    best_f = f_val
                    best_partition = partition.copy()

        if n == max_mk:
            print('-', self.f(G, PG, best_partition), self.f(G, PG, self.do_metis_with_pg(G, PG, steps_back=steps_back)))
            # sleep(3)
            print('+', self.f(G, PG, best_partition), self.f(G, PG, self.postprocessing_phase(G, PG, self.do_metis_with_pg(G, PG))))
            # sleep(3)

        return best_partition
    
    def do_MK_greed_greed(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        steps_back: int = 6,
        check_cache: bool = True,
    ) -> list[int] | None:
        print('do_MK_greed_greed')

        output_dir = output_dir.replace('results', 'results2')
        weighted_graph: nx.Graph
        if 'gen_data' in input_dir:
            weighted_graph, _, _ = input_generated_graph_and_processors_from_file(join(input_dir, graph_file))
        else:
            weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        # weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_weighted')

        start_time = time.time()
        partition = self.MK_greed_greed(weighted_graph, physical_graph, steps_back=steps_back, check_cache=check_cache)

        # assert self.f(weighted_graph, physical_graph, partition) <= self.f(weighted_graph, physical_graph, self.just_weighted_partition)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, start_time)

        return partition

    def MK_greed_greed_with_geq_cr(self, G: nx.Graph, PG: nx.Graph, check_cache: bool = True, steps_back: int = 6) -> list[int]:
        cr0 = self.CUT_RATIO

        best_best_partition: list[int] = [0] * len(G.nodes)
        best_best_f: float = self.f(G, PG, best_best_partition)

        max_mk = self.mk.get_num_mk(G, self.CUT_RATIO, steps_back=steps_back, check_cache=check_cache)

        n = 0
        if self.do_metis_with_pg(G, PG):
            n = len(set(self.do_metis_with_pg(G, PG)))
        
        if n <= max_mk:
            print('WARNING: n < MK_MAX ', n, max_mk, G.graph['graph_name'], self.CUT_RATIO)

        for cr in self.ALL_CR_LIST:
            if cr >= cr0:
                best_partition: list[int] = [0] * len(G.nodes)
                best_f: float = self.f(G, PG, best_partition)

                f: bool = False

                for nparts in range(1, max_mk + 1):
                    self.CUT_RATIO = cr
                    (G_grouped, mk_data) = self.mk.mk_nparts(G, nparts, self.CUT_RATIO, check_cache=check_cache, steps_back=steps_back)

                    print(nparts, len(set(mk_data)) if mk_data else None)

                    print('MK DATA: ', mk_data)

                    if G_grouped is None or mk_data is None:
                        continue

                    self.CUT_RATIO = 1
                    mk_partition = self.do_metis_with_pg(G_grouped, PG, check_cache=check_cache, steps_back=steps_back)
                    mk_partition = self.do_greed(G_grouped, PG, mk_partition)
                    self.CUT_RATIO = cr0

                    if mk_partition is None:    
                        continue

                    mk_partition_unpacked = unpack_mk(mk_partition, mk_data)
                    if not self.check_cut_ratio(G, mk_partition_unpacked):
                        f = True
                        break

                    if self.check_cut_ratio(G, mk_partition_unpacked):
                        print("check_passed")
                        partition = self.do_greed(G, PG, mk_partition_unpacked)
                        f_val = self.f(G, PG, partition)
                        print('f_mk_partition_unpacked: ', self.f(G, PG, mk_partition_unpacked), 'f_val: ', f_val, calc_cut_ratio(G, mk_partition_unpacked))
                        print(f_val, best_f)
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

        if n == max_mk:
            print('-', self.f(G, PG, best_partition), self.f(G, PG, self.do_metis_with_pg(G, PG)))
            print('+', self.f(G, PG, best_partition), self.f(G, PG, self.postprocessing_phase(G, PG, self.do_metis_with_pg(G, PG))))

        return best_best_partition

    def do_MK_greed_greed_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        check_cache: bool = True,
        steps_back: int = 6,
    ) -> None:
        print('do_MK_greed_greed_with_geq_cr')

        output_dir = output_dir.replace('results', 'results2')
        
        weighted_graph: nx.Graph
        if 'gen_data' in input_dir:
            weighted_graph, _, _ = input_generated_graph_and_processors_from_file(join(input_dir, graph_file))
        else:
            weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))

        # weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'MK_greed_greed_with_geq_cr')

        start_time = time.time()
        partition = self.MK_greed_greed_with_geq_cr(weighted_graph, physical_graph, check_cache=check_cache, steps_back=steps_back)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, start_time)

    def do_greed(self, G: nx.Graph, PG: nx.Graph, partition: list[int] | None) -> list[int] | None:
        print('initial partition: ', partition)
        print('f for initial partition: ', self.f(G, PG, partition))
        print('cut_ratio for initial partition: ', calc_cut_ratio(G, partition))
        partition = self.postprocessing_phase(G, PG, partition)
        print('final partition: ', partition)
        print('f for final partition: ', self.f(G, PG, partition))
        print('cut_ratio for final partition: ', calc_cut_ratio(G, partition))

        return partition

    def do_weighted_mk_with_geq_cr(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr0: float,
    ) -> None:
        print('do_mk_with_geq_cr')

        output_dir = output_dir.replace('results', 'results1')
        weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')

        print('from mk weighted')

        ans_init = None
        ans_part = None
        ans = None

        for cr in self.ALL_CR_LIST:
            if cr >= cr0:
                mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')

                if not isfile(mk_path) or not isfile(mk_data_path):
                    continue

                mk_graph_weighted = input_networkx_graph_from_file(mk_path)

                self.CUT_RATIO = 1

                initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, check_cache=True)
                assert initial_weighted_partition is not None
                weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                assert weighted_partition is not None

                initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                assert initial_weighted_partition is not None
                weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                assert weighted_partition is not None

                print('Am I even here***')

                self.CUT_RATIO = cr0
                if not self.check_cut_ratio(weighted_graph, weighted_partition):
                    print("I'm here... for some reason...")
                    break

                if ans_part is None or ans is None or self.f(weighted_graph, physical_graph, weighted_partition) < ans:
                    print('OKK????')
                    ans_init = initial_weighted_partition
                    ans_part = weighted_partition
                    ans = self.f(weighted_graph, physical_graph, weighted_partition)

        self.CUT_RATIO = cr0
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), ans_init, weighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), ans_part, weighted_graph, physical_graph)

    def simple_part(self, G: nx.Graph, PG: nx.Graph) -> list[int]:
        proc_fastest: int = 0
        speed_max: int = PG.nodes[proc_fastest]['weight']

        for proc in PG.nodes:
            speed = PG.nodes[proc]['weight']
            if speed > speed_max:
                proc_fastest = proc
                speed_max = speed

        return [proc_fastest] * len(G)

    def do_simple_part(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
    ) -> None:
        print('do_simple_part')

        output_dir = output_dir.replace('results', 'results2')
        weighted_graph: nx.Graph
        if 'gen_data' in input_dir:
            weighted_graph, _, _ = input_generated_graph_and_processors_from_file(join(input_dir, graph_file))
        else:
            weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        # weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'simple_part')
        start_time = time.time()

        partition = self.simple_part(weighted_graph, physical_graph)

        print(partition)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, start_time)

    def write_metis_with_pg(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
    ) -> None:
        print('write_metis_with_pg')

        output_dir = output_dir.replace('results', 'results2')
        weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))
        unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))

        physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'metis_with_pg')

        start_time = time.time()
        weighted_partition = self.do_metis_with_pg(weighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, start_time)
        
        start_time = time.time()
        unweighted_partition = self.do_metis_with_pg(unweighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph, start_time)

    def research(self) -> None:
        graph_dirs = [
            # (r'./data/triangle/graphs', r'./results/greed/{}triangle'),
            # (r'./data/rand', r'./results/greed/{}rand'),
            # (r'./data/testing_graphs', r'./results/greed/{}testing_graphs'),
            # (r'./data/sausages', r'./results/greed/{}sausages'),
            (r'./data/gen_data', r'./results/greed/{}gen_data'),
        ]

        physical_graph_dirs = [
            r'./data/physical_graphs',
        ]

        print('3')

        graphs = [
            # '16_envelope_mk_eq.txt',
            # '16_envelope_mk_rand.txt',
            # '64_envelope_mk_eq.txt',
            # '64_envelope_mk_rand.txt',
            # 'dag26.txt',
            # 'dag15.txt',
            # 'dag16.txt',
            # 'dag13.txt',
            # 'dag0.txt',
            # 'dagA15.txt',
            # 'dagH28.txt',
            # 'dagK43.txt',
            # 'dagN19.txt',
            # 'dagR49.txt',
            # 'triadag10_5.txt',
            # 'triadag15_4.txt',
            # 'triadag20_5.txt',
            # 'triadag25_0.txt',
            # 'triadag30_7.txt',
        ]

        graphs = [graph for graph in listdir('./data/gen_data') if 'partition' not in graph]


        # cr_list_little = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
        cr_list_little = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
        cr_list_big = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        cr_list = cr_list_little + cr_list_big
        # cr_list =  cr_list_big
        cr_list = [0.4]

        for input_dir, output_dir in graph_dirs:
            for graph_file in listdir(input_dir):
                if isfile(join(input_dir, graph_file)) and graph_file in graphs and 'partition' not in graph_file:
                    for physical_graph_dir in physical_graph_dirs:
                        for physical_graph_path in listdir(physical_graph_dir):
                            if isfile(join(physical_graph_dir, physical_graph_path)): # and '1_4x2.txt' in physical_graph_path:
                                for cr in cr_list:
                                    self.CUT_RATIO = cr
                                    weighted_graph: nx.Graph
                                    if 'gen_data' in input_dir:
                                        weighted_graph, _, _ = input_generated_graph_and_processors_from_file(join(input_dir, graph_file))
                                    else:
                                        weighted_graph = input_networkx_graph_from_file(join(input_dir, graph_file))

                                    # if len(weighted_graph.nodes) > 100:
                                        # continue

                                    if list(sorted(list(weighted_graph.nodes))) != list(range(len(weighted_graph.nodes))):
                                        continue

                                    # unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))
                                    physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))
                                    # weighted
                                    start_time = time.time()
                                    initial_weighted_partition = self.do_metis_with_pg(weighted_graph, physical_graph)
                                    self.write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph, start_time)
                                    start_time = time.time()
                                    weighted_partition = self.do_greed(weighted_graph, physical_graph, initial_weighted_partition)
                                    self.write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, start_time)

                                    # # unweighted
                                    # start_time = time.time()
                                    # initial_unweighted_partition = self.do_metis_with_pg(unweighted_graph, physical_graph)
                                    # self.write_results(join(output_dir.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph, start_time)
                                    # start_time = time.time()
                                    # unweighted_partition = self.do_greed(weighted_graph, physical_graph, initial_unweighted_partition)
                                    # self.write_results(join(output_dir.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph, start_time)

                                    # print('from mk weighted')
                                    # self.CUT_RATIO = 1
                                    # output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')
                                    # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                                    # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')
                                    # mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                                    # mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                                    # physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                                    # initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, check_cache=True)
                                    # initial_unweighted_partition = self.do_metis_with_pg(mk_graph_unweighted, physical_graph, check_cache=True)
                                    # weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                                    # unweighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                                    # initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                                    # initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                                    # weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                                    # unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                                    # self.CUT_RATIO = cr
                                    # self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                    # self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                    # print('unweighted')
                                    # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                                    # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)

                                    # print('from mk UNweighted')
                                    # self.CUT_RATIO = 1
                                    # output_dir_mk = output_dir.replace('greed', 'greed_from_mk_unweighted')
                                    # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.', str(cr) + '.')
                                    # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.txt', str(cr) + '.' + 'mapping')
                                    # mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                                    # mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                                    # physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                                    # initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, check_cache=True)
                                    # initial_unweighted_partition = self.do_metis_with_pg(mk_graph_unweighted, physical_graph, check_cache=True)
                                    # weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                                    # unweighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                                    # initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                                    # initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                                    # weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                                    # unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                                    # self.CUT_RATIO = cr
                                    # self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                                    # self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                                    # unweighted
                                    # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                                    # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)

                                    # self.do_weighted_mk_with_geq_cr(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, cr)
                                    self.do_MK_greed_greed(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, check_cache=True, steps_back=6)
                                    # self.do_MK_greed_greed_with_geq_cr(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, check_cache=True, steps_back=6)

                                    # self.do_simple_part(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path)
                                    # self.write_metis_with_pg(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path)
