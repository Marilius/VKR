from helpers import input_networkx_graph_from_file, calc_cut_ratio

from os import listdir, makedirs
from os.path import isfile, join

from random import choice, randint
import math
import json

import networkx as nx

import time


class GAP:
    def __init__(
            self,
            r1: float = 0.5,
            r2: float = 0.07,
            iter_max: int = 50,
            cut_ratio: float = 0.8,
            article: bool = False,
            graph_dirs: list[tuple[str, str]] | None = None,
            physical_graph_dirs: list[str] | None = None,
            iter_max_list: list[int] | None = None,
            r2_list: list[float] | None = None,
            cr_list: list[float] | None = None
    ) -> None:
        self.all_edges: list[tuple[int, int]]
        self.n: int
        self.k: int
        self.PG: nx.Graph
        self.G: nx.Graph
        self.R1 = r1
        self.R2 = r2
        self.ITER_MAX: int = iter_max
        self.BIG_NUMBER: float = 1e10
        self.PENALTY: bool = True
        self.CUT_RATIO: float = cut_ratio
        self.ARTICLE: bool = article

        self.NAME: str = 'GAP'
        self.MK_DIR: str = './data_mk'

        if graph_dirs:
            self.graph_dirs = graph_dirs
        else:
            self.graph_dirs = [
                (r'./data/sausages', f'./results/{self.NAME}/sausages'),
                (r'./data/triangle/graphs', f'./results/{self.NAME}/triangle'),
            ]

        if physical_graph_dirs:
            self.physical_graph_dirs = physical_graph_dirs
        else:
            self.physical_graph_dirs = [
                r'./data/physical_graphs',
            ]

        if iter_max_list:
            self.iter_max_list = iter_max_list
        else:
            self.iter_max_list = [
                20, 50, 100
            ]

        if r2_list:
            self.r2_list = r2_list
        else:
            self.r2_list = [
                0.05, 0.07, 0.1
            ]

        if cr_list:
            self.cr_list = cr_list
        else:
            self.cr_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

        if self.ARTICLE:
            self.iter_max_list = [100]
            self.r2_list = [0.07]
            # self.cr_list = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
            cr_list_little = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            cr_list_big = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            self.cr_list = cr_list_little + cr_list_big

    def calc_power(self, PG: nx.Graph) -> list[float]:
        ans: list[float] = [0] * len(PG)

        s: float = sum(data['weight'] for _, data in PG.nodes(data=True))
        for node, data in PG.nodes(data=True):
            ans[node] = data['weight'] / s

        return ans

    def get_cut_edges(self, partition: list[int]) -> list[tuple[int, int]]:
        edges = []

        for edge in self.all_edges:
            node1, node2 = edge

            if partition[node1] != partition[node2]:
                edges.append(edge)

        return edges

    def get_cut_nodes(self, cut_edges: list[tuple[int, int]]) -> list[int]:
        v_cut = set()

        for edge in cut_edges:
            (node1, node2) = edge
            v_cut.add(node1)
            v_cut.add(node2)

        v_cut = list(v_cut)
        return v_cut

    def get_edges(self, G: nx.Graph) -> list[tuple[int, int]]:
        return [edge for edge in G.edges]

    def metis_partition(self, partition_path: str, physical_graph_name: str, *args, **kwargs) -> list[int] | None:
        # HEADERS = [
        #     'graph',
        #     'physical_graph',
        #     'PENALTY',
        #     'cut_ratio',
        #     'cut_ratio_limitation',
        #     'f',
        #     'partition',
        # ]
        print(partition_path, physical_graph_name)
        print(self.PENALTY, self.CUT_RATIO)
        
        with open(partition_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # print(line)
                partition = list()
                graph, physical_graph, Penalty, cut_ratio, cut_ratio_limitation, f_val, _partition = line.split(maxsplit=6)
                if cut_ratio == 'None':
                    cut_ratio = None
                else:
                    cut_ratio = float(cut_ratio)
                Penalty = bool(Penalty)
                cut_ratio_limitation = float(cut_ratio_limitation)
                f_val = float(f_val)
                _partition = _partition.strip()
                if _partition == 'None':
                    _partition = None
                else:
                    partition = json.loads(_partition)

                if cut_ratio_limitation == self.CUT_RATIO and physical_graph == physical_graph_name and Penalty == self.PENALTY:
                    print('loaded', cut_ratio_limitation, physical_graph, 'cr:', cut_ratio)
                    print(partition_path)
                    print(partition)
                    print(f'"{_partition}"')

                    if not partition:
                        return None

                    return partition

        assert len(partition) == len(self.G), (partition_path, physical_graph_name, len(partition), len(self.G), partition)

        return partition

    def initial_partition(self, G: nx.Graph, PG: nx.Graph) -> list[int]:
        """Input: G = (V, E)
        Output: P = {P1, . . . ,Pk , Pi = (Vi, Ei), i = 1, 2, . . . , k
        1. s ← 1
        2. for i ← 1 to k do
        3.     t ← s + ri * |V| - 1
        4.     Vi ← {va | a = s,s+1, . . . ,t}
        5.     Ei ← {(va, vb) | va ∈ Vi, vb ∈ Vi}
        6.     s ← t + 1
        7. end
        8. E_cut ← {(va, vb)|va ∈ Vi, vb ∈ Vj, i != j}
        9. return P and E_cut
        """
        partition = [0] * len(G.nodes)
        r = self.calc_power(PG)

        s = 0
        for i in range(self.k):
            t = int(len(G) * r[i])

            t = max(t, 1)

            s2 = min(s + t, len(G))
            if i == len(PG) - 1:
                s2 = len(G)

            for j in range(s, s2):
                partition[j] = i

            s += t

        return partition

    def get_initial_partition(self, G: nx.Graph, PG: nx.Graph, if_do_load: bool = False, path: str | None = None, physical_graph_name: str | None = None) -> list[int] | None:
        if if_do_load and path and physical_graph_name:
            return self.metis_partition(path, physical_graph_name)
        return self.initial_partition(G, PG)

    def ga_initialization(self, cut_edges: list[tuple[int, int]]) -> list[list[int]]:
        """Input: the set of cut edges E_cut
        Output: the set of individuals X
        1. V_cut = {va|va ∈ V, ∃vb ∈ V,(va, vb) ∈ E_cut}	
        2. for i ← 1 to n do
        3.     j ← 1
        4.     while j ≤ k do
        5.         va ← a random vertex in V_cut
        6.         if a ∈/ Xi then
        7.             xij ← a
        8.             j ← j + 1
        9.         end
        10.     end
        11. end
        12. return X
        """
        v_cut = self.get_cut_nodes(cut_edges)

        self.n = int(0.4*len(cut_edges))

        individuals = []
        for _ in range(self.n):
            new_individual = []
            n_tries = 1000
            while len(new_individual) < self.k and n_tries:
                a = choice(v_cut)
                if a not in new_individual:
                    new_individual.append(a)
                else:
                    n_tries -= 1
            if n_tries == 0:
                return []
            individuals.append(new_individual)

        return individuals

    def ga_one_point_crossover(self, individuals: list[list[int]]) -> list[list[int]]:
        """Input: the set of individuals X
        Output: the updated set of individuals X
        1. for i ← 1 to trunc(n*R1) do
        2.     a ← a random number between 1 and n
        3.     b ← a random number between 1 and n
        4.     if a != b then
        5.         index ← a random number between 1 and k
        6.         for j ← 1 to index do
        7.             xnew,j ← xa,j
        8.         end
        9.         for j ← index+1 to k do
        10.            xnew,j ← xb,j
        11.        end
        12.     end
        13.     X ← X U {xnew}
        14. end
        15. return X
        """
        for _ in range(math.trunc(self.R1 * self.n)):
            new_individual = []
            a = randint(0, self.n - 1)
            b = randint(0, self.n - 1)
            if a != b:
                index = randint(0, self.k - 1)
                for j in range(index):
                    new_individual.append(individuals[a][j])
                for j in range(index, self.k):
                    new_individual.append(individuals[b][j])

                assert len(new_individual) == self.k

                individuals.append(new_individual)

        return individuals

    def ga_random_resetting(self, individuals: list[list[int]], cut_edges: list[tuple[int, int]]) -> list[list[int]]:
        """
        Input: the set of individuals X
        Output: the updated set of individuals X
        1. for i ← 1 to trunc(n*R2) do
        2.     a ← a random number between 1 and n
        3.     b ← a random number between 1 and k
        4.     vj ← a random vertex in V_cut
        5.     while j ∈ Xa do
        6.         vj ← a random vertex in V_cut
        7.     end
        8.     xa,b ← j
        9. end
        10. return X"""
        cut_nodes = self.get_cut_nodes(cut_edges)

        for _ in range(math.trunc(self.R2 * self.n)):
            a = randint(0, self.n - 1)
            b = randint(0, self.k - 1)
            v = choice(cut_nodes)

            while v in individuals[a]:
                v = choice(cut_nodes)
            individuals[a][b] = v
        return individuals

    def f(self, partition: list[int] | None, individual: list[int] | None = None) -> float:
        assert isinstance(self.G, nx.Graph)

        if partition is None:
            return 2 * self.BIG_NUMBER

        new_partition = partition.copy()

        if individual is not None:
            for j in range(self.k):
                v = individual[j]
                if v == -1:
                    continue
                new_partition[v] = j

        t_max = 0
        times = [0] * self.k
        for job, proc in enumerate(new_partition):
            times[proc] += self.G.nodes[job]['weight']

        for proc in range(self.k):
            times[proc] /= self.PG.nodes[proc]['weight']

        t_max = max(times)

        if self.PENALTY:
            cr = calc_cut_ratio(self.G, new_partition)
            if cr is None or cr > self.CUT_RATIO:
                t_max += self.BIG_NUMBER

        return t_max

    def gap(self, G: nx.Graph, PG: nx.Graph, if_do_load: bool = False, path: str | None = None, physical_graph_name: str | None = None, initial_partition: list[int] | None = None) -> list[int] | None:
        """Input: G = (V, E)
        Output: P = P1, . . . , Pk , Pi = (Vi, Ei), i = 1, 2, . . . , k
        1. Call Algorithm 1 to obtain the initial partition P
        2. fcurr ← f (G, P)
        3. flag ← false
        4. do
        5.     Call Algorithm 2 to initialize population of GA
        6.     Calculate f(Xj) for each individual Xj
        7.     f_best ← min j=1,2,...,n {f(Xj)}
        8.     Xbest ← arg min j=1,2,...,n {f(Xj)}
        9.     for i ← 1 to itermax do
        10.         Call Algorithm 3 to apply crossover
        11.         Call Algorithm 4 to apply mutation
        12.         Calculate f(Xj) for each individual Xj
        13.         f_avg ← 1/n sum from j=1 to n f(Xi)
        14.         for j ← 1 to n do
        15.             if f(Xj) ≥ f_avg then
        16.                 X ← X - {Xj}
        17.             end
        18.         end
        19.         fmin ← min j=1,2,...,n f(Xj)
        20.         Xmin ← arg min j=1,2,...,n f(Xj)
        21.         if fmin < fbest then
        22.             fbest ← fmin
        23.             Xbest ← Xmin
        24.         end
        25.     end
        26. if fbest < fcurr then
        27.     flag ← true
        28.     for j ← 1 to k do
        29.         Transfer vxbest,j to partition Pj
        30.     end
        31. end
        32. until flag = false
        """
        self.all_edges = self.get_edges(G)

        partition: list[int] | None = None
        if initial_partition is None:
            print('I WAS HERE')
            print(path, physical_graph_name, self.CUT_RATIO)
            partition = self.get_initial_partition(G, PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name)
            print('result for get_initial_partition: ', self.f(partition))
        else:
            partition = initial_partition.copy()

        if partition is None:
            return None

        print('initial_partition :', partition)
        print('result for initial_partition: ', self.f(partition))

        f_curr = self.f(partition)
        flag = True
        flag_iter = False
        epoch = 0
        while flag:
            print(f'epoch: {epoch}, f_curr: {f_curr}, current cut_ratio: {calc_cut_ratio(G, partition)}')
            flag = False

            cut_edges = self.get_cut_edges(partition)

            individuals = self.ga_initialization(cut_edges)

            ######
            if not individuals:
                break
            ####

            f_vals = [self.f(partition, i) for i in individuals[:self.n]]
            f_best = min(f_vals)
            individual_best = individuals[f_vals.index(f_best)].copy()

            generations_without_improve = 0
            while generations_without_improve < self.ITER_MAX:
                if self.k == 0:
                    return partition

                individuals = self.ga_one_point_crossover(individuals)
                individuals = self.ga_random_resetting(individuals, cut_edges)
                f_vals = [
                    self.f(partition, individuals) for individuals in individuals[:self.n]
                    ]
                f_avg = sum(f_vals) / len(f_vals)

                j = 0
                z = 0

                while j + z < self.n and individuals:
                    if len(individuals) == self.n:
                        break

                    if self.f(partition, individuals[j]) >= f_avg:
                        del individuals[j]
                        z += 1
                    j += 1

                vals = [self.f(partition, individual) for individual in individuals[:self.n]]

                assert len(vals), cut_edges

                f_min = min(vals)
                individual_min = individuals[vals.index(f_min)]

                if f_min < f_best:
                    f_best = f_min
                    individual_best = individual_min.copy()
                    generations_without_improve = 0
                else:
                    generations_without_improve += 1

            if f_best < f_curr:
                flag = True
                flag_iter = True
                for j in range(self.k):
                    v = individual_best[j]

                    partition[v] = j

                f_curr = f_best

            epoch += 1

        if flag_iter:
            print('GAP ENDED')

        return partition

    def write_results(self, path: str, physical_graph_path: str, partition: list[int] | None, start_time: float) -> None:
        # HEADERS: list[str] = [
        #     'graph',
        #     'physical_graph',
        #     'P_mut',
        #     'ITER_MAX',
        #     'cut_ratio',
        #     'Penalty',
        #     'cut_ratio_limitation',
        #     'f',
        #     'partition',
        # ]

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1],
            self.R2,
            self.ITER_MAX,
            calc_cut_ratio(self.G, partition=partition),
            self.PENALTY,
            self.CUT_RATIO,
            self.f(partition),
            partition if (cr := calc_cut_ratio(self.G, partition) is None) or cr <= self.CUT_RATIO else None,
            '\n',
        ]
        end_time = time.time()

        if partition is not None:
            assert partition, (path, partition)

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        print(f'writing results to: {path}')
        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

        overall_time = end_time - start_time

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1], 
            self.CUT_RATIO,
            start_time,
            str(overall_time),
            '\n',
        ]

        print(start_time, str(overall_time))
        print(line2write)

        with open(path.replace('.txt', '.time'), 'a+') as file:
            file.write(' '.join(map(str, line2write)))

    def research(self) -> None:
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
            'dagA15.txt',
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

        pgs = [
            # '1_4x1.txt',
            '1_4x2.txt',
            # '1_4x3.txt',
            # '1_4x4.txt',
            # '1_4x5.txt',
            # '3_2x1.txt',
            # '3_2x2.txt',
            # '3_2x3.txt',
            # '3_2x4.txt',
            # '3_2x5.txt',
            # '5_4_3_2x1.txt',
            # '5_4_3_2x2.txt',
            # '5_4_3_2x3.txt',
            # '5_4_3_2x4.txt',
            # '5_4_3_2x5.txt',
        ]

        self.cr_list = [0.4]

        for input_dir, output_dir in self.graph_dirs:
            for graph_file in listdir(input_dir):
                # print(join(input_dir, graph_file))
                if isfile(join(input_dir, graph_file)) and graph_file in graphs: # in graph_file:
                    # print(join(input_dir, graph_file))
                    for physical_graph_dir in self.physical_graph_dirs:
                        for physical_graph in listdir(physical_graph_dir):
                            if isfile(join(physical_graph_dir, physical_graph)) and physical_graph in pgs:
                                for _ in range(5):
                                    for cr in self.cr_list:
                                        self.CUT_RATIO = cr
                                        for i in self.iter_max_list:
                                            self.ITER_MAX = i
                                            for r2 in self.r2_list:
                                                self.R2 = r2
                                                # initial partition
                                                # self.G = self.input_graph_from_file(graph_path)
                                                G = input_networkx_graph_from_file(join(input_dir, graph_file))

                                                if list(sorted(list(G.nodes))) != list(range(len(G.nodes))):
                                                    continue

                                                self.G = G
                                                self.PG = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph))

                                                self.k = len(self.PG)
                                                self.all_edges = self.get_edges(self.G)
                                                # initial_partition = self.initial_partition(self.G, self.PG)
                                                # self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_initial'), join(physical_graph_dir, physical_graph), initial_partition)

                                                # partition = self.do_gap(graph_path=join(input_dir, graph_file), physical_graph_path=join(physical_graph_dir, physical_graph))
                                                # self.write_results(join(output_dir, graph_file), join(physical_graph_dir, physical_graph), partition)
                                                # from metis
                                                print('weighted_partition from metis', join(input_dir, graph_file).replace('data', 'results/metis/weighted').replace('/graphs', ''))
                                                start_time = time.time()
                                                weighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/metis/weighted').replace('/graphs', ''),
                                                    physical_graph_name=physical_graph
                                                )
                                                # assert weighted_partition, ('weighted_partition from metis', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_metis/weighted'), join(physical_graph_dir, physical_graph), weighted_partition, start_time)

                                                print('unweighted_partition from metis')
                                                start_time = time.time()
                                                unweighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/metis/unweighted').replace('/graphs', ''),
                                                    physical_graph_name=physical_graph
                                                )
                                                # assert unweighted_partition, ('unweighted_partition from metis', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_metis/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition, start_time)

                                                # from greed
                                                print('weighted_partition from greed')
                                                start_time = time.time()
                                                weighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/greed/weighted').replace('/graphs', ''),
                                                    physical_graph_name=physical_graph
                                                )
                                                # assert weighted_partition, ('weighted_partition from greed', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_greed/weighted'), join(physical_graph_dir, physical_graph), weighted_partition, start_time)

                                                print('unweighted_partition from greed')
                                                start_time = time.time()
                                                unweighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/greed/unweighted').replace('/graphs', ''),
                                                    physical_graph_name=physical_graph
                                                )
                                                # assert unweighted_partition, ('unweighted_partition from greed', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_greed/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition, start_time)

                                                # FROM MK
                                                # print('partition FROM WEIGHTED MK')
                                                # self.CUT_RATIO = 1
                                                # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                                                # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')

                                                # weighted_partition = self.do_gap(
                                                    # graph_path=mk_path,
                                                    # physical_graph_path=join(physical_graph_dir, physical_graph),
                                                # )
                                                # self.CUT_RATIO = cr
                                                # if weighted_partition is not None:
                                                    # weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                                                # assert weighted_partition, ('weighted_partition FROM WEIGHTED MK', mk_path)
                                                # self.G = G
                                                # self.all_edges = self.get_edges(G)
                                                # self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_from_mk/weighted'), join(physical_graph_dir, physical_graph), weighted_partition)

                                                # print('partition FROM UNWEIGHTED MK')
                                                # self.CUT_RATIO = 1
                                                # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.', str(cr) + '.')
                                                # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.txt', str(cr) + '.' + 'mapping')

                                                # unweighted_partition = self.do_gap(
                                                    # graph_path=mk_path,
                                                    # physical_graph_path=join(physical_graph_dir, physical_graph),
                                                # )
                                                # print(self.f(unweighted_partition), '//////////before unpack////////////')
                                                # self.CUT_RATIO = cr
                                                # print(unweighted_partition)
                                                # if unweighted_partition is not None:
                                                    # unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)
                                                # print(unweighted_partition)

                                                # assert unweighted_partition, ('unweighted_partition FROM UNWEIGHTED MK', mk_path)
                                                # self.G = G
                                                # self.all_edges = self.get_edges(G)
                                                # print(self.f(unweighted_partition), '//////////AFTER unpack////////////')
                                                # self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_from_mk/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition)
                                                # print(G)
                                                # print(calc_cut_ratio(self.G, unweighted_partition), cr)
                                                # print(mk_path)
                                                # if cr == 0.2:
                                                #   raise Exception

    def do_gap(self, graph_path: str, physical_graph_path: str, if_do_load: bool = False, path: str | None = None, physical_graph_name: str | None = None) -> list[int] | None:
        print(graph_path, physical_graph_path)

        self.G: nx.Graph = input_networkx_graph_from_file(graph_path)
        self.PG: nx.Graph = input_networkx_graph_from_file(physical_graph_path)

        self.k = len(self.PG)

        return self.gap(self.G, self.PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name)
