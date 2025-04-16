import settings as settings

from algo.helpers import calc_edgecut, check_cut_ratio, f, add_cache_check

from os import makedirs
from os.path import isfile

import networkx as nx
import metis


class BasePartitioner:
    @add_cache_check
    def metis_part(self, G: nx.Graph, nparts: int, ufactor: int, seed: int | None, recursive: bool | None = True, ) -> list[int]:
        """
        Use METIS to partition a graph.

        Args:
            G (nx.Graph): The graph to be partitioned.
            nparts (int): The number of parts to partition the graph into.
            ufactor (int): The imbalance factor for METIS.
            check_cache (bool): Whether to check the cache before calling METIS.
            seed (int | None): The seed for METIS.
            recursive (bool | None): Whether to use the recursive partitioning method of METIS.

        Returns:
            tuple[int, list[int]]: A tuple containing the edgecut and a list of the partition
            assignment for each node.
        """
        if recursive is None:
            recursive = nparts > 8

        if nparts == 1:
            return [0] * len(G.nodes)

        (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive, seed=seed)
        assert len(partition2parse) == len(G.nodes)

        partition = [0] * len(G.nodes)

        for new_i, i in enumerate(list(G.nodes)):
            partition[i] = partition2parse[new_i]

        for new_i, i in enumerate(sorted(list(set(partition)))):
            for j in range(len(partition)):
                if partition[j] == i:
                    partition[j] = new_i

        assert len(partition) == len(G.nodes)

        return partition
    
    def load_do_metis_cache(self, G: nx.Graph, nparts: int, recursive: bool, cr_max: float, steps_back: int) -> list[int] | None:
        """
        Loads the result of a do_metis function call from the cache for a given graph configuration.

        Args:
            G (nx.Graph): The graph for which the partition is loaded.
            nparts (int): The number of parts the graph was partitioned into.
            recursive (bool): Whether the recursive partitioning method was used.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            list[int] | None: The partition assignment for each node if it exists in the cache, otherwise None.
        """

        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/base_do_metis/{G_hash}_{nparts}_{cr_max}_{recursive}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_do_metis_cache(self, G: nx.Graph, nparts: int, recursive: bool, cr_max: float, partition: list[int] | None, steps_back: int) -> None:
        """
        Writes the result of the do_metis function call to the cache for a given graph.

        This function stores the partitioning result in a cache file, allowing for
        quick retrieval in future runs if the same input parameters are used.

        Args:
            G (nx.Graph): The graph for which the partition is cached.
            nparts (int): The number of parts the graph was partitioned into.
            recursive (bool): Whether the recursive partitioning method was used.
            cr_max (float): The maximum allowed cut ratio.
            partition (list[int] | None): The partition assignment for each node.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            None
        """

        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/base_do_metis/{G_hash}_{nparts}_{cr_max}_{recursive}_{steps_back}.txt'

        # weighted = '_w_' if 'node_weight_attr' in G.graph else '_!'
        # path = settings.CACHE_DIR + '/' + G.graph['graph_name'] + weighted + str(nparts) + '!_' + str(cr_max) + '_' + str(recursive) + f'_{steps_back}_' + '.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def do_metis(self, G: nx.Graph, nparts: int, cr_max: float, check_cache: bool, seed: int | None, recursive: bool | None = None, steps_back: int = 5, ) -> list[int] | None:
        """
        Use METIS to partition a graph with a given maximum cut ratio.

        Args:
            G (nx.Graph): The graph to be partitioned.
            nparts (int): The number of parts to partition the graph into.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before calling METIS.
            seed (int | None): The seed for METIS.
            recursive (bool | None): Whether to use the recursive partitioning method of METIS.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            list[int] | None: A list of the partition assignment for each node if the cut ratio is
            satisfied, otherwise None.
        """
        if G is None or nparts is None:
            raise TypeError()

        if recursive is None:
            recursive = nparts > 8

        if nparts == 1:
            return [0] * len(G)

        if check_cache:
            partition = self.load_do_metis_cache(G, nparts, recursive, cr_max, steps_back=steps_back)
            if partition and len(partition) == len(G.nodes) and check_cut_ratio(G, partition, cr_max):
                return partition

        ufactor = 1

        partition = self.metis_part(G=G, nparts=nparts, ufactor=ufactor, seed=seed, recursive=recursive, check_cache=check_cache)

        while not check_cut_ratio(G, partition, cr_max):
            ufactor *= 2

            if ufactor > 10e7:
                return None

            partition = self.metis_part(G=G, nparts=nparts, ufactor=ufactor, seed=seed, recursive=recursive, check_cache=check_cache)

        ans = partition.copy()
        for _ in range(steps_back):
            ufactor *= 0.75
            ufactor = int(ufactor)
            if ufactor < 1:
                break

            partition = self.metis_part(G=G, nparts=nparts, ufactor=ufactor, seed=seed, recursive=recursive, check_cache=check_cache)
            assert len(partition) == len(G.nodes), (f'len(partition): {len(partition)}, len(G.nodes): {len(G.nodes)}', ufactor)
            if check_cut_ratio(G, partition, cr_max):
                ans = partition.copy()

        if check_cache:
            self.write_do_metis_cache(G, nparts, recursive, cr_max, ans, steps_back)

        return ans

    def load_do_metis_with_pg_cache(self, G: nx.Graph, PG: nx.Graph, cr_max: float, steps_back: int = 5) -> list[int] | None:
        """
        Loads result of do_metis_with_pg function from the cache for a given graph, physical graph, the maximum allowed cut ratio and the number of steps back.

        Args:
            G (nx.Graph): The graph.
            PG (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int, optional): The number of times of ufactor reducing. Defaults to 5.

        Returns:
            list[int] | None: The result of do_metis_with_pg function call if it exists in the cache, otherwise None.
        """
        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        PG_hash = nx.weisfeiler_lehman_graph_hash(PG, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/do_metis_with_pg/{G_hash}_{PG_hash}_{cr_max}_{steps_back}.txt'

        if isfile(path):
            with open(path, 'r') as f:
                line = f.readline()
                if 'None' not in line:
                    partition = list(map(int, line.split()))

                    print('CACHED! :)')

                    return partition
        
        return None

    def write_do_metis_with_pg_cache(self, G: nx.Graph, PG: nx.Graph, cr_max: float, steps_back: int, partition: list[int] | None) -> None:
        """
        Writes the result of the do_metis_with_pg function call to the cache for a given graph and physical graph.

        This function stores the partitioning result in a cache file, allowing for
        quick retrieval in future runs if the same input parameters are used.

        Args:
            G (nx.Graph): The graph for which the partition is cached.
            PG (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.
            steps_back (int): The number of times of ufactor reducing.
            partition (list[int] | None): The partition assignment for each node.

        Returns:
            None
        """

        node_attr = 'weight' if 'node_weight_attr' in G.graph else None
        G_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr=node_attr)
        PG_hash = nx.weisfeiler_lehman_graph_hash(PG, node_attr=node_attr)
        path = f'{settings.CACHE_DIR}/do_metis_with_pg/{G_hash}_{PG_hash}_{cr_max}_{steps_back}.txt'

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'w') as file:
            if partition:
                file.write(' '.join(map(str, partition)))
            else:
                file.write('None')

    def do_metis_with_pg(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, seed: int | None, steps_back: int = 7) -> list[int] | None:
        """
        Runs the METIS partitioning algorithm on a given graph with a given number of parts and maximum allowed cut ratio, then
        assign then assigns groups of tasks to processors according to their performance.

        Args:
            G (nx.Graph): The graph to be partitioned.
            PG (nx.Graph): The physical graph.
            cr_max (float): The maximum allowed cut ratio.
            check_cache (bool): Whether to check the cache before running the algorithm.
            seed (int | None): The seed for the algorithm.
            steps_back (int): The number of times of ufactor reducing.

        Returns:
            list[int] | None: The best partition found, or None if no valid partition was found.
        """
        if check_cache:
            partition = self.load_do_metis_with_pg_cache(G, PG, cr_max, steps_back=steps_back)
            if partition and len(partition) == len(G) and check_cut_ratio(G, partition, cr_max):
                return partition

        partition = self.do_metis(G, len(PG), cr_max, check_cache, seed, steps_back=steps_back)

        procs = [i for i in PG.nodes]
        procs = list(sorted(procs, key=lambda proc: -PG.nodes[proc]['weight']))
        f_1 = f(G, PG, partition, cr_max)
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

        f_2 = f(G, PG, partition, cr_max)
        assert f_2 <= f_1, (f_2, f_1)

        if check_cache:
            self.write_do_metis_with_pg_cache(G, PG, cr_max, steps_back, partition)

        return partition
