from helpers import input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file, calc_edgecut, calc_cut_ratio, do_unpack_mk, unpack_mk

from os import listdir, makedirs
from os.path import isfile, join

from time import sleep

import networkx as nx
import metis

class BasePartitioner:
    def __init__(self) -> None:
        self.MK_DIR: str = './data_mk'
        self.CACHE_DIR: str = './cache'
        self.ALL_CR_LIST: list[float] = [i/100 for i in range(7, 100)] + [1]
        self.BIG_NUMBER: float = 1e10
        self.PENALTY: bool = True
        self.CUT_RATIO: float = 0.7

    @staticmethod
    def metis_part(G: nx.Graph, nparts: int, ufactor: int, recursive: bool) -> tuple[int, list[int]]:
        # print('here')
        if nparts == 1:
            return (0, [0] * len(G.nodes))

        (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)
        # print('not here')

        partition = [0] * len(G.nodes)

        print(partition, partition2parse)
        print(list(G.nodes))

        print(*enumerate(list(G.nodes)))

        for new_i, i in enumerate(list(G.nodes)):
            partition[i] = partition2parse[new_i]

        for new_i, i in enumerate(sorted(list(set(partition)))):
            for j in range(len(partition)):
                if partition[j] == i:
                    partition[j] = new_i

        assert edgecuts == calc_edgecut(G, partition)

        return (edgecuts, partition)

    def check_cut_ratio(self, G: nx.Graph | None, partition: list[int] | None) -> bool:
        if G is None or partition is None:
            return False

        return calc_cut_ratio(G, partition) <= self.CUT_RATIO  # type: ignore
    
    def f(self, G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None) -> float:
        p_loads = [0] * len(PG)

        if G is None or partition is None:
            return 2 * self.BIG_NUMBER

        for i in range(len(partition)):
            p_loads[partition[i]] += G.nodes[i]['weight']

        for i in range(len(PG)):
            p_loads[i] /= PG.nodes[i]['weight'] 

        penalty = 0

        if self.PENALTY:
            penalty = 0 if self.check_cut_ratio(G, partition) else self.BIG_NUMBER

        return max(p_loads) + penalty
    
    def load_metis_cache(self, G: nx.Graph, PG: nx.Graph, recursive: bool) -> list[int] | None:
        nparts: int
        if isinstance(PG, nx.Graph):
            nparts = len(PG)
        else:
            nparts = PG

        weighted = '_weighted_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + weighted + str(nparts) + '!_' + str(self.CUT_RATIO) + '_' + str(recursive) + '.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_metis_cache(self, G: nx.Graph, PG: nx.Graph | int, recursive: bool, partition: list[int] | None) -> None:
        nparts: int
        if isinstance(PG, nx.Graph):
            nparts = len(PG)
        else:
            nparts = PG
        
        weighted = '_w_' if 'node_weight_attr' in G.graph else '_!'
        path = self.CACHE_DIR + '/' + G.graph['graph_name'] + weighted + str(nparts) + '!_' + str(self.CUT_RATIO) + '_' + str(recursive) + '.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def do_metis(self, G: nx.Graph | None, PG: nx.Graph | None, recursive: bool | None = None, check_cache: bool = True) -> list[int] | None:
        if G is None or PG is None:
            return None

        # if isinstance(PG, int):
            # PG = self.get_homo_pg(PG)

        if recursive is None:
            if (len(PG) > 8):
                recursive = True
            else:
                recursive = False

        if check_cache:
            partition = self.load_metis_cache(G, PG, recursive)
            if partition and len(partition) == len(G) and self.check_cut_ratio(G, partition):
                return partition

        if len(PG) == 1:
            return [0] * len(G)

        ufactor = 1

        (_, partition) = self.metis_part(G, len(PG), ufactor, recursive)

        while not self.check_cut_ratio(G, partition):
            ufactor *= 2

            if ufactor > 10e7:
                return None

            (_, partition) = self.metis_part(G, len(PG), ufactor, recursive)

        ans = partition.copy()
        for _ in range(5):
            ufactor *= 0.75
            ufactor = int(ufactor)
            if ufactor < 1:
                break

            (_, partition) = self.metis_part(G, len(PG), ufactor, recursive)
            if self.check_cut_ratio(G, partition):
                # if self.f(G, PG, partition) < self.f(G, PG, ans):
                ans = partition

        if check_cache:
            self.write_metis_cache(G, PG, recursive, ans)

        return ans
    
    @staticmethod
    def get_homo_pg(nproc: int) -> nx.Graph:
        physical_graph = nx.complete_graph(nproc)

        for i in range(nproc):
            physical_graph.nodes[i]['weight'] = 1

        physical_graph.graph['graph_name'] = 'homo_' + str(nproc) + '.txt'
        physical_graph.graph['node_weight_attr'] = 'weight'

        return physical_graph
