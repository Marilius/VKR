from os import listdir
from os.path import isfile, join

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from random import choice, randint
import math


all_edges = None
n = None
k = None
# PG = None
# G = None
R1 = 0.5
R2 = 0.07
ITER_MAX = 50
BIG_NUMBER = 1e10
PENALTY = True
CUT_RATIO = 0.8


@dataclass
class Node:
    size: float
    children: list[str]


def input_graph_from_file(path: str) -> dict[int: Node]:
    graph = dict()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, size, *children = map(float, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            if name not in graph:
                graph[name] = Node(size, children)
    return graph


def calc_power(PG: dict[int: Node]) -> list[float]:
    ans = []
    s = sum(node.size for node in PG.values())
    for node in PG.values():
        ans.append(node.size / s)

    return ans


def calc_edgecut(partition: dict[int, list[int]]) -> int:
    global all_edges
    edgecut = 0

    for edge in all_edges:
        node1, node2 = edge

        for i in partition:
            if node1 in partition[i] and node2 in partition[i]:
                break
        else:
            edgecut += 1

    return edgecut


def calc_cut_ratio(partition: dict[int, list[int]]) -> float:
    global all_edges

    if len(all_edges):
        return calc_edgecut(partition) / len(all_edges)

    return 0


def get_cut_edges(partition: dict[int, list[int]]) -> list[tuple[int, int]]:
    global all_edges
    edges = []

    for edge in all_edges:
        node1, node2 = edge

        for i in partition:
            if node1 in partition[i] and node2 in partition[i]:
                break
        else:
            edges.append(edge)

    return edges


def get_cut_nodes(cut_edges: list[tuple[int, int]]) -> list[int]:
    global G
    v_cut = set()

    for node1 in G:
        for node2 in G[node1].children:
            if (node1, node2) in cut_edges:
                v_cut.add(node1)

    # print(v_cut)
    v_cut = list(v_cut)
    return v_cut


def get_edges(G: dict[int: Node]) -> list[tuple[int, int]]:
    edges = []
    for node in G:
        for node2 in G[node].children:
            edges.append((node, node2))
    return edges


def initial_partition(G: dict[int: Node], PG: dict[int: Node]) -> dict[int, list[int]]:
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
    global k
    partition = defaultdict(list)
    r = calc_power(PG)

    s = 0
    for i in range(k):
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


def ga_initialization(G: dict[int: Node], cut_edges: list[tuple[int, int]]) -> list[list[int]]:
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
    global n, k

    v_cut = get_cut_nodes(cut_edges)

    n = int(0.4*len(cut_edges))

    individuals = []
    for i in range(n):
        new_individual = []
        while len(new_individual) < k:
            a = choice(v_cut)
            if a not in new_individual:
                new_individual.append(a)
        individuals.append(new_individual)

    return individuals


def ga_one_point_crossover(individuals: list[list[int]]) -> list[list[int]]:
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
    global n, k

    for i in range(math.trunc(R1*n)):
        new_individual = []
        a = randint(0, n - 1)
        b = randint(0, n - 1)
        if a != b:
            index = randint(0, k - 1)
            for j in range(index):
                new_individual.append(individuals[a][j])
            for j in range(index, k):
                new_individual.append(individuals[b][j])

            assert len(new_individual) == k

            individuals.append(new_individual)
    
    return individuals


def ga_random_resetting(individuals: list[list[int]], cut_edges: list[tuple[int, int]]) -> list[list[int]]:
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
    global n, k

    cut_nodes = get_cut_nodes(cut_edges)

    for i in range(math.trunc(R2*n)):
        a = randint(0, n - 1)
        b = randint(0, k - 1)
        v = choice(cut_nodes)
        # --------------------------------------------------------
        while v in individuals[a]:
            v = choice(cut_nodes)
        individuals[a][b] = v
    return individuals


def f(partition: dict[int, list[int]] | None, individual: list[int] = None) -> float:
    global G, PG, all_edges, k

    if partition is None:
        return 2 * BIG_NUMBER

    new_partition = deepcopy(partition)

    if individual is not None:
        for j in range(k):
            v = individual[j]
            for z in range(k):
                if v in new_partition[z]:
                    new_partition[z].remove(v)
            new_partition[j].append(v)

    t_max = 0
    for proc in new_partition:
        t_curr = 0

        for node in new_partition[proc]:
            t_curr += G[node].size
        t_curr /= PG[proc].size

        if t_curr > t_max:
            t_max = t_curr

    if PENALTY:
        if calc_cut_ratio(new_partition) > CUT_RATIO:
            t_max += BIG_NUMBER

    return t_max


def gap(G: dict[int: Node], PG: dict[int: Node]) -> dict[int, list[int]]:
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
    global all_edges, n, R1
    all_edges = get_edges(G)

    partition = initial_partition(G, PG)

    print('initial_partition :', partition)
    print('result for initial_partition: ', f(partition))

    f_curr = f(partition)
    flag = True
    flag_iter = False
    epoch = 0
    while flag:
        print(f'epoch: {epoch}, f_curr: {f_curr}')
        flag = False

        cut_edges = get_cut_edges(partition)

        individuals = ga_initialization(G, cut_edges)

        ######
        if not individuals:
            break
        ####

        f_vals = [f(partition, i) for i in individuals[:n]]
        # f_vals = [f(i) for i in individuals] # до n ???? - вероятно авторы хотели так
        f_best = min(f_vals)
        individual_best = individuals[f_vals.index(f_best)].copy()

        for i in range(ITER_MAX):
            if k == 0:
                return partition

            individuals = ga_one_point_crossover(individuals)
            individuals = ga_random_resetting(individuals, cut_edges)
            f_vals = [f(partition, individuals)
                      for individuals in individuals[:n]]
            f_avg = sum(f_vals) / len(f_vals)

            j = 0
            z = 0

            # костыльно, но иначе удаляется больше половины индивидов
            # как будто можно переписать чище
            while j + z < n and individuals:
                if len(individuals) == n:
                    break

                if f(partition, individuals[j]) >= f_avg:
                    del individuals[j]
                    z += 1
                j += 1

            vals = [f(partition, individual) for individual in individuals[:n]]

            assert len(vals), cut_edges

            f_min = min(vals)
            individual_min = individuals[vals.index(f_min)]

            if f_min < f_best:
                f_best = f_min
                individual_best = individual_min.copy()

        if f_best < f_curr:
            flag = True
            flag_iter = True
            for j in range(k):
                v = individual_best[j]

                for z in range(k):
                    if v in partition[z]:
                        partition[z].remove(v)

                partition[j].append(v)

            f_curr = f_best

        epoch += 1

    if flag_iter:
        print('GAP ENDED')

    return partition


def flatten_partition(partition: dict[int: list[int]]) -> list[int]:
    ...


def write_results(path: str, physical_graph_path: str, partition: dict[int: list[int]]) -> None:
    global R2, ITER_MAX, PENALTY, CUT_RATIO

    HEADERS: list[str] = [
        'graph',
        'physical_graph',
        'P_mut',
        'ITER_MAX',
        'cut_ratio',
        'Penalty',
        'cut_ratio_limitation',
        'f',
        'partition',
    ]

    line2write = [
        path.split('/')[-1],
        physical_graph_path.split('/')[-1],
        R2,
        ITER_MAX,
        calc_cut_ratio(partition=partition),
        PENALTY,
        CUT_RATIO,
        f(partition),
        dict(partition),
        '\n',
    ]

    with open(path, 'a+') as file:
        file.write(' '.join(map(str, line2write)))

    # print(' '.join(map(str, line2write)))


def research() -> None:
    global ITER_MAX, R2, CUT_RATIO
    graph_dirs = [
        (r'../data/sausages', r'../results/GAP/sausages'),
        (r'../data/triangle/graphs', r'../results/GAP/triangle'),
    ]

    physical_graph_dirs = [
        r'../data/physical_graphs',
    ]   

    iter_max_list = [
        20, 50, 100
    ]

    r2_list = [
        0.05, 0.07, 0.1
    ]

    print('3')


    for input_dir, output_dir in graph_dirs:
        for graph_file in listdir(input_dir):
            # print(join(input_dir, graph_file))
            if isfile(join(input_dir, graph_file)):
                # print(join(input_dir, graph_file))
                for physical_graph_dir in physical_graph_dirs:
                    for physical_graph in listdir(physical_graph_dir):
                        if isfile(join(physical_graph_dir, physical_graph)):
                            for _ in range(5):
                                for cr in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                                    CUT_RATIO = cr
                                    for i in iter_max_list:
                                        ITER_MAX = i
                                        for r2 in r2_list:
                                            R2 = r2
                                            partition = do_gap(graph_path=join(input_dir, graph_file), physical_graph_path=join(physical_graph_dir, physical_graph))
                                            write_results(join(output_dir, graph_file), join(physical_graph_dir, physical_graph), partition)


def do_gap(graph_path: str, physical_graph_path: str) -> dict[int: list[int]]:
    global k, G, PG

    print(graph_path, physical_graph_path)

    G = input_graph_from_file(graph_path)
    PG = input_graph_from_file(physical_graph_path)

    k = len(PG)

    return gap(G, PG)


if __name__ == '__main__':
    # необходимо чтобы граф был связный
    # достаточно(для запуска алгоритма), чтобы было верно следующее:
    # (minimum_cut/2)*R0 >= кол-во процессоров
    # minimum_cut/2 (точнее количество вершин, которые принадлежат этим рёбрам)
    # ещё лучше, если значение выражения больше желаемого размера популяции
    
    # G = input_graph_from_file(r'../data/triangle/graphs/triadag10_0.txt')
    # G = input_graph_from_file(r'../data/sausages/dagA0.txt')
    # G = input_graph_from_file(r'../data/trash/graph100.txt')
    # G = input_graph_from_file(r'../data/trash/graph10.txt')
    # G = input_graph_from_file(r'../data/trash/test1.txt')
    # G = input_graph_from_file(r'../data/trash/gap2.txt')
    # G = input_graph_from_file(r'../data/trash/hetero0.txt')
    # G = input_graph_from_file(r'../data/trash/hetero1.txt')

    # PG = input_graph_from_file(r'../data/test_gp/0.txt')
    # PG = input_graph_from_file(r'../data/test_gp/homo3.txt')

    # print(G)
    # print(PG)

    # k = len(PG)
    # print(k)

    # R1 = 1
    # R2 = 0.15
    # ITER_MAX = 100

    # partition = gap(G, PG)

    # e = []
    # for proc in partition:
    #     for node in partition[proc]:
    #         e.append(node)

    # assert len(e) == len(G)
    # assert len(set(e)) == len(set(G))

    # print('final partition :', partition)
    # print('result for final partition: ', f(partition))

    # print('cut_edges:', get_cut_edges(partition))

    # print('cut_ratio:', calc_cut_ratio(partition))

    research()
