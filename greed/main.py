from base_partitioner.main import BasePartitioner

from helpers import input_graph, input_networkx_unweighted_graph_from_file, calc_cut_ratio


from os import makedirs
from os.path import join

import networkx as nx

import time


class GreedPartitioner(BasePartitioner):
    def write_results(self, path: str, physical_graph_path: str, partition: list[int], G: nx.Graph, PG: nx.Graph, cr_max: float, start_time: float) -> None:
        # HEADERS: list[str] = [
        #     'graph',
        #     'physical_graph',
        #     'cut_ratio',
        #     'cut_ratio_limitation',
        #     'f',
        #     'partition',
        # ]

        end_time = time.time()

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1],
            calc_cut_ratio(G=G, partition=partition),
            cr_max,
            self.f(G, PG, partition, cr_max),
            partition if self.check_cut_ratio(G, partition, cr_max) else None,
            '\n',
        ]

        assert partition is None or len(set(partition)) <= len(PG.nodes)

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

        overall_time = end_time - start_time

        line2write = [
            path.split('/')[-1],
            physical_graph_path.split('/')[-1],
            cr_max,
            len(partition) if partition is not None else None,
            start_time,
            str(overall_time),
            '\n',
        ]

        with open(path.replace('.txt', '.time'), 'a+') as file:
            file.write(' '.join(map(str, line2write)))

    def postprocessing_phase(self, G: nx.Graph | None, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> list[int] | None:
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
                            if self.check_cut_ratio(G, partition_copy, cr_max):
                                p_loads[proc] += a_weight
                                p_loads[p1] -= a_weight
                                partition[a] = proc
                                flag = True
                                break

                if flag:
                    break

        return partition

    def do_greed(self, G: nx.Graph, PG: nx.Graph, partition: list[int] | None, cr_max: float) -> list[int] | None:
        if partition is None:
            return None

        print('BASE', 'cr:', calc_cut_ratio(G, partition))
        weights = [0] * len(PG)
        for i in range(len(partition)):
            weights[partition[i]] += G.nodes[i]['weight']
        print('BASE', weights)
        print(self.f(G, PG, partition, cr_max))

        partition = self.postprocessing_phase(G, PG, partition, cr_max)
        print('GREED', 'cr:', calc_cut_ratio(G, partition))
        weights = [0] * len(PG)
        for i in range(len(partition)):
            weights[partition[i]] += G.nodes[i]['weight']
        print('GREED', weights)
        print(self.f(G, PG, partition, cr_max))

        return partition

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
        output_dir = output_dir.replace('results', 'results2')
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'simple_part')
        start_time = time.time()

        partition = self.simple_part(weighted_graph, physical_graph)

        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), partition, weighted_graph, physical_graph, start_time)

    def write_metis_with_pg(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
    ) -> None:
        output_dir = output_dir.replace('results', 'results2')
        weighted_graph = input_graph(join(input_dir, graph_file))
        unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))

        physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
        output_dir_mk = output_dir.replace('greed', 'metis_with_pg')

        start_time = time.time()
        weighted_partition = self.do_metis_with_pg(weighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, start_time)

        start_time = time.time()
        unweighted_partition = self.do_metis_with_pg(unweighted_graph, physical_graph)
        self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph, start_time)

    def run(self, graph: nx.Graph, physical_graph: nx.Graph, cr_max: float) -> list[int] | None:
        initial_weighted_partition = self.do_metis_with_pg(graph, physical_graph, cr_max)
        
        partition = self.postprocessing_phase(graph, physical_graph, initial_weighted_partition, cr_max)

        return partition
