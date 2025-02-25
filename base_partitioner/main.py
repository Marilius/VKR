import base_partitioner.settings as settings

from helpers import calc_edgecut, calc_cut_ratio

from os import makedirs
from os.path import isfile

import networkx as nx
import metis


class BasePartitioner:
    def load_metis_part_cache(self, G: nx.Graph, nparts: int, ufactor: int, recursive: bool) -> list[int] | None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/metis_part/{G_hash}_{nparts}_{ufactor}_{recursive}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                partition = list(map(int, line.split()))
                return partition
        
        return None

    def write_metis_part_cache(self, G: nx.Graph, nparts: int, ufactor: int, recursive: bool, partition: list[int]) -> None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/metis_part/{G_hash}_{nparts}_{ufactor}_{recursive}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            file.write(' '.join(map(str, partition)))

    def metis_part(self, G: nx.Graph, nparts: int, ufactor: int, check_cache: bool, seed: int | None, recursive: bool | None = True, ) -> tuple[int, list[int]]:
        if recursive is None:
            recursive = nparts > 8

        if nparts == 1:
            return (0, [0] * len(G.nodes))

        if check_cache:
            partition = self.load_metis_part_cache(G, nparts, ufactor, recursive)
            if partition:
                return (calc_edgecut(G, partition), partition)

        (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive, seed=seed)
        assert len(partition2parse) == len(G.nodes)

        partition = [0] * len(G.nodes)

        for new_i, i in enumerate(list(G.nodes)):
            print('new_i, i', new_i, i)
            print(len(partition2parse), len(G.nodes))
            partition[i] = partition2parse[new_i]

        for new_i, i in enumerate(sorted(list(set(partition)))):
            for j in range(len(partition)):
                if partition[j] == i:
                    partition[j] = new_i

        if check_cache:
            self.write_metis_part_cache(G, nparts, ufactor, recursive, partition)

        assert len(partition) == len(G.nodes)

        return (edgecuts, partition)

    def check_cut_ratio(self, G: nx.Graph | None, partition: list[int] | None, cr_max: float) -> bool:
        if G is None or partition is None:
            return False

        return calc_cut_ratio(G, partition) <= cr_max  # type: ignore
    
    def f(self, G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> float:
        p_loads = [0] * len(PG)

        if G is None or partition is None:
            return 2 * settings.BIG_NUMBER

        for i in range(len(partition)):
            p_loads[partition[i]] += G.nodes[i]['weight']

        for i in range(len(PG)):
            p_loads[i] /= PG.nodes[i]['weight'] 

        penalty = 0 if self.check_cut_ratio(G, partition, cr_max) else settings.BIG_NUMBER

        return max(p_loads) + penalty
    
    def load_do_metis_cache(self, G: nx.Graph, nparts: int, recursive: bool, cr_max: float, steps_back: int) -> list[int] | None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/base_do_metis/{G_hash}_{nparts}_{cr_max}_{recursive}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_do_metis_cache(self, G: nx.Graph, nparts: int, recursive: bool, cr_max: float, partition: list[int] | None, steps_back: int) -> None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/base_do_metis/{G_hash}_{nparts}_{cr_max}_{recursive}_{steps_back}.txt'

        # weighted = '_w_' if 'node_weight_attr' in G.graph else '_!'
        # path = settings.CACHE_DIR + '/' + G.graph['graph_name'] + weighted + str(nparts) + '!_' + str(cr_max) + '_' + str(recursive) + f'_{steps_back}_' + '.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def do_metis(self, G: nx.Graph, nparts: int, cr_max: float, check_cache: bool, seed: int | None, recursive: bool | None = None, steps_back: int = 5, ) -> list[int] | None:
        if G is None or nparts is None:
            raise TypeError()

        if recursive is None:
            recursive = nparts > 8

        if nparts == 1:
            return [0] * len(G)

        if check_cache:
            partition = self.load_do_metis_cache(G, nparts, recursive, cr_max, steps_back=steps_back)
            if partition and len(partition) == len(G.nodes) and self.check_cut_ratio(G, partition, cr_max):
                return partition

        ufactor = 1

        (_, partition) = self.metis_part(G, nparts, ufactor, check_cache, seed, recursive)
        print('kikikiki', len(G.nodes), len(partition))

        while not self.check_cut_ratio(G, partition, cr_max):
            ufactor *= 2

            if ufactor > 10e7:
                return None

            (_, partition) = self.metis_part(G, nparts, ufactor, check_cache, seed, recursive)

        # print(len(set(partition)))
        # print('kakakakak', len(G.nodes), len(partition))
        # print('nparts', nparts)

        ans = partition.copy()
        for _ in range(steps_back):
            ufactor *= 0.75
            ufactor = int(ufactor)
            if ufactor < 1:
                break

            print(f'len(partition): {len(partition)}, len(G.nodes): {len(G.nodes)}', ufactor)
            (_, partition) = self.metis_part(G, nparts, ufactor, check_cache, seed, recursive)
            assert len(partition) == len(G.nodes), (f'len(partition): {len(partition)}, len(G.nodes): {len(G.nodes)}', ufactor)
            if self.check_cut_ratio(G, partition, cr_max):
                ans = partition.copy()

        # print('kukukukuk', len(G.nodes), len(partition))
        if check_cache:
            self.write_do_metis_cache(G, nparts, recursive, cr_max, ans, steps_back)

        # print('ans', len(G.nodes), len(ans))

        return ans

    def load_do_metis_with_pg_cache(self, G: nx.Graph, PG: nx.Graph, cr_max: float, steps_back: int = 5) -> list[int] | None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        PG_hash = nx.weisfeiler_lehman_graph_hash(PG, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/do_metis_with_pg/{G_hash}_{PG_hash}_{cr_max}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_do_metis_with_pg_cache(self, G: nx.Graph, PG: nx.Graph, cr_max: float, steps_back: int, partition: list[int] | None) -> None:
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        PG_hash = nx.weisfeiler_lehman_graph_hash(PG, node_attr=node_attr)
        path = f'2{settings.CACHE_DIR}/do_metis_with_pg/{G_hash}_{PG_hash}_{cr_max}_{steps_back}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def do_metis_with_pg(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 7) -> list[int] | None:
        if check_cache:
            partition = self.load_do_metis_with_pg_cache(G, PG, cr_max, steps_back=steps_back)
            if partition and len(partition) == len(G) and self.check_cut_ratio(G, partition, cr_max):
                return partition

        partition = self.do_metis(G, len(PG), cr_max, check_cache, seed, steps_back=steps_back)

        procs = [i for i in PG.nodes]
        procs = list(sorted(procs, key=lambda proc: -PG.nodes[proc]['weight']))
        f_1 = self.f(G, PG, partition, cr_max)
        if partition is not None:
            times = [0] * len(PG)
            for i, proc in enumerate(partition):
                times[proc] += G.nodes[i]['weight']
            procs_old = [i for i in PG.nodes]
            procs_old = list(sorted(procs_old, key=lambda proc: -times[proc]))
            
            for i in range(len(partition)):
                partition[i] = procs[procs_old.index(partition[i])]

            times = [0] * len(PG)
            for i, proc in enumerate(partition):
                times[proc] += G.nodes[i]['weight']

        f_2 = self.f(G, PG, partition, cr_max)
        assert f_2 <= f_1, (f_2, f_1)

        if check_cache:
            self.write_do_metis_with_pg_cache(G, PG, cr_max, steps_back, partition)

        return partition
