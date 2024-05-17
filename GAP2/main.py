from helpers import input_networkx_graph_from_file, calc_cut_ratio, do_unpack_mk

from GAP import GAP

from random import randint, choice

import networkx as nx

import math


class GAP2(GAP):
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
            cr_list: list[float] | None = None,
            num_of_tries: int = 20
    ) -> None:
        super().__init__(r1, r2, iter_max, cut_ratio, article)
        self.NAME = 'GAP2'
        self.MK_DIR = './data_mk'
        self.NUM_OF_TRIES = num_of_tries

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

    def ga_initialization(self, cut_edges: list[tuple[int, int]]) -> list[list[int]]:
        v_cut = self.get_cut_nodes(cut_edges)

        # if not self.n:
        self.n = int(0.4*len(cut_edges))
        # self.n = int(0.6*len(cut_edges))

        if self.n <= 1:
            self.n = 1

        individuals = []
        for _ in range(self.n):
            new_individual = []
            n_tries = 1000
            while len(new_individual) < self.k and n_tries:
                a = choice(v_cut)
                a = choice(v_cut + [-1])
                if a not in new_individual:
                    new_individual.append(a)
                n_tries -= 1
            if n_tries == 0:
                return []
            individuals.append(new_individual)

        return individuals

    def ga_random_resetting(self, individuals: list[list[int]], cut_edges: list[tuple[int, int]]) -> list[list[int]]:
        cut_nodes = self.get_cut_nodes(cut_edges)

        for _ in range(math.trunc(self.R2 * self.n)):
            a = randint(0, self.n - 1)
            b = randint(0, self.k - 1)
            v = choice(cut_nodes + [-1])
            # --------------------------------------------------------
            while v in individuals[a]:
                v = choice(cut_nodes + [-1])
            individuals[a][b] = v
        return individuals

    def f(self, partition: list[int] | None, individual: list[int] | None = None, black_list: list[dict] | None = None) -> float:
        if partition and black_list and individual:
            partition2 = partition.copy()

            for j in range(self.k):
                v = individual[j]
                if v == -1:
                    continue

                partition2[v] = j

            if partition2 in black_list:
                return 2 * self.BIG_NUMBER

        penalty = 0

        return super().f(partition, individual) + penalty

    def gap(self, G: nx.Graph, PG: nx.Graph, if_do_load: bool = False, path: str | None = None, physical_graph_name: str | None = None, initial_partition: list[int] | None = None) -> list[int]:
        black_list = []

        num_of_tries = self.NUM_OF_TRIES

        print('was there')

        self.all_edges = self.get_edges(G)

        partition = None
        if not initial_partition:
            partition = self.get_initial_partition(G, PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name)
        else:
            partition = initial_partition

        ans = partition.copy()
        f_ans = self.f(partition)

        print('initial_partition :', partition)
        print('result for initial_partition: ', self.f(partition))
        print('cut ratio for initial_partition: ', calc_cut_ratio(G, partition))

        while num_of_tries:
            print('number of tries left:', num_of_tries)
            print(partition)
            # sleep(0.5)
            f_curr = self.f(partition, black_list=black_list)
            epoch = 0
            flag = True
            flag_iter = False
            individuals = None
            while flag:
                print(f'epoch: {epoch}, f_ans: {f_ans}, f_curr: {f_curr}, current cut_ratio: {calc_cut_ratio(G, partition)}')
                flag = False

                cut_edges = self.get_cut_edges(partition)

                #########
                if not cut_edges:
                    break
                ########

                individuals = self.ga_initialization(cut_edges)

                print(len(cut_edges), self.n)

                ######
                if not individuals:
                    break
                ####

                f_vals = [self.f(partition, i, black_list=black_list) for i in individuals[:self.n]]
                # print(f_vals)
                f_best = min(f_vals)
                individual_best = individuals[f_vals.index(f_best)].copy()

                for _ in range(self.ITER_MAX):
                    if self.k == 0:
                        return partition

                    individuals = self.ga_one_point_crossover(individuals)
                    individuals = self.ga_random_resetting(individuals, cut_edges)
                    f_vals = [
                        self.f(partition, individuals, black_list=black_list) for individuals in individuals[:self.n]
                    ]

                    f_avg = sum(f_vals) / len(f_vals)

                    j = 0
                    z = 0

                    # костыльно, но иначе удаляется больше половины индивидов
                    # как будто можно переписать чище
                    while j + z < self.n and individuals:
                        if len(individuals) == self.n:
                            break

                        if self.f(partition, individuals[j], black_list=black_list) >= f_avg:
                            del individuals[j]
                            z += 1
                        j += 1

                    vals = [self.f(partition, individual, black_list=black_list) for individual in individuals[:self.n]]

                    assert len(vals), cut_edges

                    f_min = min(vals)
                    individual_min = individuals[vals.index(f_min)]

                    if f_min < f_best:
                        f_best = f_min
                        individual_best = individual_min.copy()

                if f_best < f_curr:
                    # ??????????/
                    black_list.append(partition.copy())
                    # ????????????????
                    flag = True
                    flag_iter = True
                    for j in range(self.k):
                        v = individual_best[j]
                        if v == -1:
                            continue

                        partition[v] = j

                    f_curr = f_best

                if f_curr < f_ans:
                    num_of_tries = self.NUM_OF_TRIES
                    ans = partition.copy()
                    f_ans = f_curr

                epoch += 1
            # else:
                # print('SUKAAAA')

            if f_curr < f_ans:
                ans = partition.copy()
                f_ans = f_curr
            cut_edges = self.get_cut_edges(ans)

            if not cut_edges:
                break

            individuals = self.ga_initialization(cut_edges)

            # f_vals = [self.f(partition, individuals, black_list=black_list)
            #         for individuals in individuals]
            # print(f_vals)

            if individuals:
                individual = individuals[randint(0, len(individuals) - 1)]

                black_list.append(ans.copy())

                # a = randint(0, len(individuals) - 1)
                # f_vals = [self.f(partition, i, black_list=black_list) for i in individuals]
                # print(f_vals)
                # f_best = min(f_vals)
                # individual_best = individuals[f_vals.index(f_best)].copy()

                for j in range(self.k):
                    v = individual[j]
                    if v == -1:
                        continue

                    partition[v] = j

                f_curr = f_best
            num_of_tries -= 1

        print('final f result = ', f_ans)
        print('final partition = ', ans)
        print('f(ans) = ', self.f(ans))
        if flag_iter:
            print(f'{self.NAME} ENDED')

        assert len(ans) == len(G), (path, physical_graph_name)
        check2 = [0] * len(PG)
        for job, proc in enumerate(ans):
            # for job in ans[k]:
            check2[proc] += G.nodes[job]['weight']
        print('check2 = ', check2)
        for k in PG:
            check2[k] /= PG.nodes[k]['weight']
        print('check2 = ', check2)

        return ans
