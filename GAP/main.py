from os import listdir, makedirs
from os.path import isfile, join

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from random import choice, randint
import math
import json


@dataclass
class Node:
    size: float
    children: list[str]

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
        self.all_edges = None
        self.n = None
        self.k = None
        self.PG = None
        self.G = None
        self.R1 = r1
        self.R2 = r2
        self.ITER_MAX = iter_max
        self.BIG_NUMBER = 1e10
        self.PENALTY = True
        self.CUT_RATIO = cut_ratio
        self.ARTICLE = article

        self.NAME = 'GAP'
        self.MK_DIR = '../data_mk'

        if graph_dirs:
            self.graph_dirs = graph_dirs
        else:
            self.graph_dirs = [
                (r'../data/sausages', f'../results/{self.NAME}/sausages'),
                (r'../data/triangle/graphs', f'../results/{self.NAME}/triangle'),
            ]

        if physical_graph_dirs:
            self.physical_graph_dirs = physical_graph_dirs
        else:
            self.physical_graph_dirs = [
                r'../data/physical_graphs',
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
            self.iter_max_list = [50]
            self.r2_list = [0.07]
            self.cr_list = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]


    def unpack_mk(self, mk_partition: dict[int: list[int]], mk_data: list[int]) -> dict[int: list[int]]:
        # print(mk_partition)
        # print(mk_data)

        ans = dict()
        # mapping = dict()

        # for mk_id, proc in enumerate(mk_partition):
            # mapping[mk_id] = proc

        # for i in range(len(mk_data)):
            # ans[i] = mapping[ans[i]]

        # print(mk_partition, '<----')
        for proc, mk_ids in mk_partition.items():
            # print(proc, mk_ids)
            ans[proc] = []
            for job, mk in enumerate(mk_data):
                if mk in mk_ids:
                    ans[proc].append(job)

        return ans


    def do_unpack_mk(self, mk_partition: dict[int: list[int]], mk_data_path: str) -> dict[int: list[int]]:
        with open(mk_data_path, 'r') as file:
            line = file.readline()
            mk_data = list(map(int, line.split()))
            print(mk_data, '!!!!!!!!!!!!!!!!!mk_data!!!!!!!!!!!!!!!!!!!!!!')

            return self.unpack_mk(mk_partition, mk_data)


    def input_graph_from_file(self, path: str) -> dict[int: Node]:
        graph = dict()
        with open(path, 'r') as f:
            for line in f.readlines()[1:]:
                name, size, *children = map(float, line.strip().split())
                name = int(name)
                children = list(map(int, children))
                if name not in graph:
                    graph[name] = Node(size, children)
        return graph


    def calc_power(self, PG: dict[int: Node]) -> list[float]:
        ans = []
        s = sum(node.size for node in PG.values())
        for node in PG.values():
            ans.append(node.size / s)

        return ans


    def calc_edgecut(self, partition: dict[int, list[int]]) -> int:
        edgecut = 0

        for edge in self.all_edges:
            node1, node2 = edge

            for i in partition:
                if node1 in partition[i] and node2 in partition[i]:
                    break
            else:
                edgecut += 1

        return edgecut


    def calc_cut_ratio(self, partition: dict[int, list[int]]) -> float:
        if len(self.all_edges):
            return self.calc_edgecut(partition) / len(self.all_edges)

        return 0


    def get_cut_edges(self, partition: dict[int, list[int]]) -> list[tuple[int, int]]:
        edges = []

        for edge in self.all_edges:
            node1, node2 = edge

            for i in partition:
                if node1 in partition[i] and node2 in partition[i]:
                    break
            else:
                edges.append(edge)

        return edges


    def get_cut_nodes(self, cut_edges: list[tuple[int, int]]) -> list[int]:
        v_cut = set()

        for node1 in self.G:
            for node2 in self.G[node1].children:
                if (node1, node2) in cut_edges:
                    v_cut.add(node1)

        # print(v_cut)
        v_cut = list(v_cut)
        return v_cut


    def get_edges(self, G: dict[int: Node]) -> list[tuple[int, int]]:
        edges = []
        for node in G:
            for node2 in G[node].children:
                edges.append((node, node2))
        return edges


    def metis_partition(self, partition_path: str, physical_graph_name: str, *args, **kwargs) -> dict[int, list[int]]:
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
        partition = defaultdict(list)
        with open(partition_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line)
                graph, physical_graph, Penalty, cut_ratio, cut_ratio_limitation, f_val, _partition = line.split(maxsplit=6)
                cut_ratio = float(cut_ratio)
                Penalty = bool(Penalty)
                cut_ratio_limitation = float(cut_ratio_limitation)
                f_val = float(f_val)
                _partition = json.loads(_partition)

                print(physical_graph)
                print(graph)


                print('------', partition_path, line, '-----------')
                if graph in partition_path and physical_graph in physical_graph_name and self.PENALTY == Penalty and self.CUT_RATIO == cut_ratio_limitation:
                    for job, proc in enumerate(_partition):
                        partition[proc].append(job)
                    break
        print(partition)
        print(self.f(partition))

        print(partition)

        assert len(self.flatten_partition(partition)) == len(self.G), (partition_path, physical_graph_name, len(self.flatten_partition(partition)), len(self.G), partition)

        # raise(Exception)

        return partition


    def initial_partition(self, G: dict[int: Node], PG: dict[int: Node]) -> dict[int, list[int]]:
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
        partition = defaultdict(list)
        r = self.calc_power(PG)

        s = 0
        for i in range(self.k):
            t = int(len(G) * r[i])
            # print(i, t)

            # костыль, которого в статье не было: 
            t = max(t, 1)
            # -------------------- но артефакт забавный - если мощность процессора слишком маленькая/мало вершин в графе, то на него не назначается ни одной работы 

            s2 = min(s + t, len(G))
            if i == len(PG) - 1:
                s2 = len(G)

            for j in range(s, s2):
                partition[i].append(j)

            s += t

        return partition


    def get_initial_partition(self, G: dict[int: Node], PG: dict[int: Node], if_do_load: bool = False, path: str = None, physical_graph_name: str = None) -> dict[int, list[int]]:
        if if_do_load:
            return self.metis_partition(path, physical_graph_name)
        return self.initial_partition(G, PG)


    def ga_initialization(self, G: dict[int: Node], cut_edges: list[tuple[int, int]]) -> list[list[int]]:
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

        # if not self.n:
        self.n = int(0.4*len(cut_edges))

        individuals = []
        for _ in range(self.n):
            new_individual = []
            n_tries = 1000
            while len(new_individual) < self.k and n_tries:
                a = choice(v_cut)
                if a not in new_individual:
                    new_individual.append(a)
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
            # --------------------------------------------------------
            while v in individuals[a]:
                v = choice(cut_nodes)
            individuals[a][b] = v
        return individuals


    def f(self, partition: dict[int, list[int]] | None, individual: list[int] = None) -> float:
        if partition is None:
            return 2 * self.BIG_NUMBER

        new_partition = deepcopy(partition)

        if individual is not None:
            for j in range(self.k):
                v = individual[j]
                if v == -1:
                    continue
                for z in range(self.k):
                    if v in new_partition[z]:
                        new_partition[z].remove(v)
                new_partition[j].append(v)

        t_max = 0
        for proc in new_partition:
            t_curr = 0

            for node in new_partition[proc]:
                t_curr += self.G[node].size
            t_curr /= self.PG[proc].size

            if t_curr > t_max:
                t_max = t_curr

        if self.PENALTY:
            if self.calc_cut_ratio(new_partition) > self.CUT_RATIO:
                t_max += self.BIG_NUMBER

        return t_max


    def gap(self, G: dict[int: Node], PG: dict[int: Node], if_do_load: bool = False, path: str = None, physical_graph_name: str = None, initial_partition: dict[int, list[int]] | None = None) -> dict[int, list[int]]:
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

        partition = None
        if initial_partition is None:
            partition = self.get_initial_partition(G, PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name)
        else:
            partition = deepcopy(initial_partition)

        print('initial_partition :', partition)
        print('result for initial_partition: ', self.f(partition))

        f_curr = self.f(partition)
        flag = True
        flag_iter = False
        epoch = 0
        while flag:
            print(f'epoch: {epoch}, f_curr: {f_curr}, current cut_ratio: {self.calc_cut_ratio(partition)}')
            flag = False

            cut_edges = self.get_cut_edges(partition)

            individuals = self.ga_initialization(G, cut_edges)

            ######
            if not individuals:
                break
            ####

            f_vals = [self.f(partition, i) for i in individuals[:self.n]]
            # f_vals = [f(i) for i in individuals] # до n ???? - вероятно авторы хотели так
            f_best = min(f_vals)
            individual_best = individuals[f_vals.index(f_best)].copy()

            for _ in range(self.ITER_MAX):
                if self.k == 0:
                    return partition

                individuals = self.ga_one_point_crossover(individuals)
                individuals = self.ga_random_resetting(individuals, cut_edges)
                f_vals = [self.f(partition, individuals)
                        for individuals in individuals[:self.n]]
                f_avg = sum(f_vals) / len(f_vals)

                j = 0
                z = 0

                # костыльно, но иначе удаляется больше половины индивидов
                # как будто можно переписать чище
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

            if f_best < f_curr:
                flag = True
                flag_iter = True
                for j in range(self.k):
                    v = individual_best[j]

                    for z in range(self.k):
                        if v in partition[z]:
                            partition[z].remove(v)

                    partition[j].append(v)

                f_curr = f_best

            epoch += 1

        if flag_iter:
            print('GAP ENDED')

        return partition


    def flatten_partition(self, partition: dict[int: list[int]]) -> list[int]:
        flat_partition = []
        n = len(self.G)
        for i in range(n):
            for proc in partition:
                if i in partition[proc]:
                    flat_partition.append(proc)
        return flat_partition


    def write_results(self, path: str, physical_graph_path: str, partition: dict[int: list[int]]) -> None:
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
            self.calc_cut_ratio(partition=partition),
            self.PENALTY,
            self.CUT_RATIO,
            self.f(partition),
            self.flatten_partition(dict(partition)) if self.calc_cut_ratio(partition) <= self.CUT_RATIO else None,
            '\n',
        ]

        if partition is not None:
            assert partition, (path, partition)

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        print(f'writing results to: {path}')
        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))


    def research(self) -> None:
        for input_dir, output_dir in self.graph_dirs:
            for graph_file in listdir(input_dir):
                # print(join(input_dir, graph_file))
                if isfile(join(input_dir, graph_file)): #'K46' in graph_file: # in graph_file:
                    # print(join(input_dir, graph_file))
                    for physical_graph_dir in self.physical_graph_dirs:
                        for physical_graph in listdir(physical_graph_dir):
                            if isfile(join(physical_graph_dir, physical_graph)) and '3_2x1correct.txt' in physical_graph:
                                for _ in range(5):
                                    for cr in self.cr_list:
                                        self.CUT_RATIO = cr
                                        for i in self.iter_max_list:
                                            self.ITER_MAX = i
                                            for r2 in self.r2_list:
                                                self.R2 = r2
                                                # initial partition
                                                # self.G = self.input_graph_from_file(graph_path)
                                                G = self.input_graph_from_file(join(input_dir, graph_file))

                                                self.G = G
                                                self.PG = self.input_graph_from_file(join(physical_graph_dir, physical_graph))

                                                self.k = len(self.PG)
                                                self.all_edges = self.get_edges(self.G)
                                                initial_partition = self.initial_partition(self.G, self.PG)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_initial'), join(physical_graph_dir, physical_graph), initial_partition)

                                                partition = self.do_gap(graph_path=join(input_dir, graph_file), physical_graph_path=join(physical_graph_dir, physical_graph))
                                                self.write_results(join(output_dir, graph_file), join(physical_graph_dir, physical_graph), partition)
                                                # from metis
                                                print('weighted_partition from metis')
                                                weighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/metis/weighted'),
                                                    physical_graph_name=physical_graph
                                                )
                                                assert weighted_partition, ('weighted_partition from metis', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_metis/weighted'), join(physical_graph_dir, physical_graph), weighted_partition)

                                                print('unweighted_partition from metis')
                                                unweighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/metis/unweighted'),
                                                    physical_graph_name=physical_graph
                                                )
                                                assert unweighted_partition, ('unweighted_partition from metis', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_metis/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition)

                                                # from greed
                                                print('weighted_partition from greed')
                                                weighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/greed/weighted'),
                                                    physical_graph_name=physical_graph
                                                )
                                                assert weighted_partition, ('weighted_partition from greed', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_greed/weighted'), join(physical_graph_dir, physical_graph), weighted_partition)

                                                print('unweighted_partition from greed')
                                                unweighted_partition = self.do_gap(
                                                    graph_path=join(input_dir, graph_file),
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                    if_do_load=True,
                                                    path=join(input_dir, graph_file).replace('data', 'results/greed/unweighted'),
                                                    physical_graph_name=physical_graph
                                                )
                                                assert unweighted_partition, ('unweighted_partition from greed', join(input_dir, graph_file),)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_greed/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition)

                                                # FROM MK
                                                print('partition FROM WEIGHTED MK')
                                                self.CUT_RATIO = 1
                                                mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                                                mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')

                                                weighted_partition = self.do_gap(
                                                    graph_path=mk_path,
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                )
                                                self.CUT_RATIO = cr
                                                weighted_partition = self.do_unpack_mk(weighted_partition, mk_data_path)
                                                assert weighted_partition, ('weighted_partition FROM WEIGHTED MK', mk_path)
                                                self.G = G
                                                self.all_edges = self.get_edges(G)
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_from_mk/weighted'), join(physical_graph_dir, physical_graph), weighted_partition)

                                                print('partition FROM UNWEIGHTED MK')
                                                self.CUT_RATIO = 1
                                                mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.', str(cr) + '.')
                                                mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.txt', str(cr) + '.' + 'mapping')

                                                unweighted_partition = self.do_gap(
                                                    graph_path=mk_path,
                                                    physical_graph_path=join(physical_graph_dir, physical_graph),
                                                )
                                                print(self.f(unweighted_partition), '//////////before unpack////////////')
                                                self.CUT_RATIO = cr
                                                # print(unweighted_partition)
                                                unweighted_partition = self.do_unpack_mk(unweighted_partition, mk_data_path)
                                                # print(unweighted_partition)

                                                assert unweighted_partition, ('unweighted_partition FROM UNWEIGHTED MK', mk_path)
                                                self.G = G
                                                self.all_edges = self.get_edges(G)
                                                print(self.f(unweighted_partition), '//////////AFTER unpack////////////')
                                                self.write_results(join(output_dir, graph_file).replace(self.NAME, f'{self.NAME}_from_mk/unweighted'), join(physical_graph_dir, physical_graph), unweighted_partition)
                                                print(G)
                                                print(self.calc_cut_ratio(unweighted_partition), cr)
                                                print(mk_path)
                                                # if cr == 0.2:
                                                    # raise Exception
                                                

    def do_gap(self, graph_path: str, physical_graph_path: str, if_do_load: bool = False, path: str = None, physical_graph_name: str = None) -> dict[int: list[int]]:
        print(graph_path, physical_graph_path)

        self.G = self.input_graph_from_file(graph_path)
        self.PG = self.input_graph_from_file(physical_graph_path)

        self.k = len(self.PG)

        return self.gap(self.G, self.PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name)


if __name__ == '__main__':
    graph_dirs = [
        (r'../data/testing_graphs', '../results/GAP/testing_graphs'),
        (r'../data/triangle/graphs', '../results/GAP/triangle'),
        (r'../data/sausages', '../results/GAP/sausages'),
    ]
    gap_alg = GAP(article=True, graph_dirs=graph_dirs)
    # необходимо чтобы граф был связный
    # достаточно(для запуска алгоритма), чтобы было верно следующее:
    # (minimum_cut/2)*R0 >= кол-во процессоров
    # minimum_cut/2 (точнее количество вершин, которые принадлежат этим рёбрам)
    # ещё лучше, если значение выражения больше желаемого размера популяции

    # graph_name = r'../data/sausages/dagA0.txt'
    # physical_graph_name = r'../data/physical_graphs/1234x3.txt'
    # G = input_graph_from_file(graph_name)
    # PG = input_graph_from_file(physical_graph_name)    

    # physical_graph_dirs = [
    #     r'../data/physical_graphs',
    # ]

    gap_alg.research()
