import math
from copy import deepcopy
from dataclasses import dataclass
from random import choice, randint

R1 = 0.5
R2 = 0.07
G = None
PG = None
all_edges = None
BIG_NUMBER = 1e10
PENALTY = True
CUT_RATIO = 0.45


@dataclass
class Node:
    id: int
    weight: float
    children: list[str]


@dataclass
class Dendrogram:
    id: int
    weight: float
    data: Node | None = None
    parent: int | None = None
    leftchild: int | None = None
    rightchild: int | None = None


def get_edges(G: dict[int: Node]) -> list[list[int, int]]:
    edges = []
    for node in G:
        for node2 in G[node].children:
            edges.append([node, node2])
    return edges


def input_graph_from_file(path: str) -> dict[int: Node]:
    graph = dict()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            name, size, *children = map(float, line.strip().split())
            name = int(name)
            children = list(map(int, children))
            if name not in graph:
                graph[name] = Node(name, size, children)
    return graph


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


def phase1(G: dict[int: Node]) -> Dendrogram:
    """Phase1()
    Input: task graph
    1. Initialize each node as a single element cluster
    2. Sort edges of task graph in ascending order of their weights
    3. While not all nodes in one cluster
    4.     take edge next in order
    5.     determine groups u and v to which the end nodes of the current edge belongs
    6.     m = merge (u,v)
    7.     add to dendrogram(a,u,v)
    8. end while
    """
    # Initialize each node as a single element cluster
    # index = G[len(G) - 1].id + len(G) + 1
    index = max(G.keys())
    G_copy = deepcopy(G)
    dendrogram = []
    for node in G:
        dendrogram.append(
            Dendrogram(G[node].id, G[node].weight, G[node])
        )

        index += 1

    # Sort edges of task graph in ascending order of their weights
    edges = get_edges(G)  # all weights = 1, soooo

    # While not all nodes in one cluster
    while len(dendrogram) > 1:
        # take edge next in order
        edge = edges.pop(0)
        # determine groups u and v to which the end nodes of the current edge belongs
        u, v = edge
        if u == v:
            continue

        # print(dendrogram)
        # print(edges)

        for i in range(len(dendrogram)):
            # print(i)
            if dendrogram[i].id == u:
                u = dendrogram[i]
            if dendrogram[i].id == v:
                v = dendrogram[i]

        # u, v = dendrogram[u], dendrogram[v]

        assert isinstance(u, Dendrogram) and isinstance(
            v, Dendrogram), ((u, v), dendrogram)

        m = Dendrogram(
            index,
            u.weight + v.weight,
            leftchild=u.id,
            rightchild=v.id
        )

        u.parent = index
        v.parent = index

        for edge in edges:
            if edge[0] == u.id or edge[0] == v.id:
                edge[0] = index
            elif edge[1] == u.id or edge[1] == v.id:
                edge[1] = index

        index += 1

        G[u.id] = u
        G[v.id] = v

        dendrogram.append(m)
        dendrogram.remove(u)
        dendrogram.remove(v)

    print(dendrogram)

    return dendrogram[0]


def phase2(dendrogram: Dendrogram, resource_clusters):
    """Phase2()
    Input: dendrogram, resource clusters
    1. task clusters[1] = dendrogram[root]
    2. traverse dendrogram()
    3. GA(task clusters, resource clusters)
    4. Send task clusters to their respective resource clusters
    """
    task_clusters = []
    task_clusters.append(dendrogram)
    ...


def traverse_dendrogram():
    """traverse dendrogram()
    1. While n ≤ number of resource clusters
    2.     task clusters [n++] = dendrogram[root].rchild
    3.     task clusters [n] = dendrogram[root].lchild
    4.     If n ≤ number of clusters
    5.         n--
    6.         traverse dendrogram()
    7.     End if
    8. End while
    """
    ...


def ga_two_point_crossover(individuals0: list[list[int]]) -> list[list[int]]:
    individuals = deepcopy(individuals0)
    
    # TODO проверить можно ли убрать
    n = len(individuals)
    k = len(individuals[0])

    for _ in range(math.trunc(R1*n)):
        new_individual1 = []
        new_individual2 = []

        a = randint(0, n - 1)
        b = randint(0, n - 1)
        if a != b:
            index = randint(0, k - 1)
            for j in range(index):
                new_individual1.append(individuals[a][j])
                new_individual2.append(individuals[b][j])
            for j in range(index, k):
                new_individual1.append(individuals[b][j])
                new_individual2.append(individuals[a][j])

            assert len(new_individual1) == k
            assert len(new_individual2) == k

            individuals.append(new_individual1)
            individuals.append(new_individual2)
    
    return individuals


def ga_mutation(individuals0: list[list[int]]) -> list[list[int]]:
    individuals = deepcopy(individuals0)
    
    # TODO проверить можно ли убрать
    n = len(individuals)
    k = len(individuals[0])

    for _ in range(math.trunc(R2*n)):
        index = randint(0, n - 1)

        a = randint(0, k - 1)
        b = randint(0, k - 1)
        
        individuals[index][a], individuals[index][b] = individuals[index][b], individuals[index][a]
    
    return individuals


if __name__ == '__main__':
    # G = input_graph_from_file(r'../data/triangle/graphs/triadag10_0.txt')
    # G = input_graph_from_file(r'../data/sausages/dagA0.txt')
    # G = input_graph_from_file(r'../data/trash/graph100.txt')
    # G = input_graph_from_file(r'../data/trash/graph10.txt')
    G = input_graph_from_file(r'../data/trash/gap1.txt')
    PG = input_graph_from_file(r'../data/test_gp/0.txt')

    k = len(PG)
    all_edges = get_edges(G)

    # print(G)

    phase1(G)

    # partition = gap(G, PG)

    # e = []
    # for proc in partition:
    #     for node in partition[proc]:
    #         e.append(node)

    # assert len(e) == len(G)
    # assert len(set(e)) == len(set(G))

    # print(partition)
    # print(f(partition))
    # print('cut_ratio:', calc_cut_ratio(partition))

    # print(PG)
