from greed import f, input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file, write_results, calc_edgecut, calc_cut_ratio

import networkx as nx
import metis

import os

from collections import defaultdict


class MK:
    def __init__(self, data_dirs: list[str]) -> None:
        self.CUT_RATIO = 0.3
        self.BIG_NUMBER = 1e10
        self.output_folder = './data_mk/{}'

        self.data_dirs = data_dirs


    def check_cut_ratio(self, G: nx.Graph, partition: list[int]) -> bool:
        return calc_cut_ratio(G, partition) <= self.CUT_RATIO


    def write_mk(self, g_name: str, G_grouped: nx.Graph, mk_partition: list[int]) -> None:
        output_file = self.output_folder.format(g_name.replace('.', str(self.CUT_RATIO) + '.'))

        # print(output_file, G_grouped)
        # print(sorted(list(G_grouped.nodes)))

        with open(output_file, 'w+') as file:
            # if os.path.getsize(self.output_folder.format(g_name)) == 0:
            file.write('name weight children\n')
            
            for node_id in sorted(list(G_grouped.nodes)):
                line = [str(node_id), str(G_grouped.nodes[node_id]['weight'])]

                for neighbor in G_grouped.neighbors(node_id):
                    if neighbor > node_id:
                        line.append(str(neighbor)) 

                line.append('\n')
                # print(f"'{line}'")
                file.write(' '.join(line))

        output_file = self.output_folder.format(g_name.replace('.txt', str(self.CUT_RATIO) + '.' + 'mapping'))
        with open(output_file, 'w+') as file:
            file.write(' '.join(map(str, mk_partition)))


    def do_metis(self, G: nx.Graph, nparts: int, ufactor: int, recursive: bool | None = None) -> list[int]:
        if recursive is None:
            if (nparts > 8):
                recursive = True
            else:
                recursive = False

        if nparts == 1: # Floating point exception from metis ¯\_(ツ)_/¯
            return [0] * len(G.nodes)

        (edgecuts, partition2parce) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)
        
        partition = [0] * len(G.nodes)
        
        for new_i, i in enumerate(list(G.nodes)):
            partition[i] = partition2parce[new_i]

        assert edgecuts == calc_edgecut(G, partition)
        assert len(partition) == len(G.nodes)

        return partition


    def mk_part(self, G: nx.Graph, PG: nx.Graph) -> tuple[int, list[int]]:
        num_left = len(PG)
        num_right = len(G)

        n_ans = None
        partition_ans = None
        
        n = 0
        ufactor = 0

        while num_right - num_left > 1:
            # print('--------')
            # print(num_left, num_right)
            n = (num_left + num_right) // 2
            print(n)

            ufactor = 1
            while True:
                partition = self.do_metis(G, n, ufactor)
                print(partition)
                print(calc_cut_ratio(G, partition))

                print(self.check_cut_ratio(G, partition))

                if self.check_cut_ratio(G, partition):
                    print('was here')
                    if len(set(partition)) == n:
                        num_left = n
                        n_ans = n
                        partition_ans = partition
                    else:
                        num_right = n

                    if n_ans is None or len(set(partition)) > n_ans:
                        num_left = len(set(partition)) - 1
                        n_ans = len(set(partition))
                        partition_ans = partition

                    break

                if ufactor > 100000:
                    print('ENDED BY UFACTOR')
                    num_right = n - 1
                    break

                # TODO как менять ufactor?
                ufactor += ufactor
            # print('--------')
    
        print('main ended')

        partition = self.do_metis(G, num_right, ufactor)
        print(partition)
        print(calc_cut_ratio(G, partition))

        if self.check_cut_ratio(G, partition):
            if n_ans is None or len(set(partition)) > n_ans:
                n_ans = num_right
                partition_ans = partition

        partition = self.do_metis(G, num_left, ufactor)
        print(partition)
        print(calc_cut_ratio(G, partition))

        if self.check_cut_ratio(G, partition):
            if n_ans is None or len(set(partition)) > n_ans:
                n_ans = num_left
                partition_ans = partition

        assert partition_ans, partition_ans

        if set(range(n_ans)) != set(partition_ans):
            mapping = dict()
            
            # print(partition_ans)

            for new_id, old_id in enumerate(set(partition_ans)):
                mapping[old_id] = new_id

            for i in range(len(partition_ans)):
                partition_ans[i] = mapping[partition_ans[i]]

            # print(partition_ans)

            # raise Exception

        return (n_ans, partition_ans)
    

    # def mk_part(self, G: nx.Graph, PG: nx.Graph) -> tuple[int, list[int]]:
    #     num_left = len(PG)
    #     num_right = len(G)

    #     n_ans = None
    #     partition_ans = None
        
    #     n = 0
    #     ufactor = 0

    #     for n in range(num_right, num_left, - 1):
    #         ufactor = 1000
    #         while True:
    #             partition = self.do_metis(G, n, ufactor)
    #             print(ufactor, n)
    #             print(partition)

    #             if self.check_cut_ratio(G, partition):
    #                 if len(set(partition)) == n:
    #                     n_ans = n
    #                     partition_ans = partition
    #                     break

    #             if ufactor > 10000:
    #                 print('ENDED BY UFACTOR')
    #                 break

    #             ufactor += ufactor

    #         if n_ans == n:
    #             break
            
    #         # TODO как менять ufactor?
            
    #         # print('--------')

    #     return (n_ans, partition_ans)


    def group_mk(self, G: nx.Graph, partition: list[int], weighted: bool = True) -> nx.Graph:
        grouped_G = nx.Graph()
        nodes_ids = sorted(list(set(partition)))

        for node_id in nodes_ids:
            weight = 0
            for num, part in enumerate(partition):
                if part == node_id:
                    weight += G.nodes[num]['weight']
            grouped_G.add_node(node_id, weight=weight)

        for old_node_id, node_id in enumerate(partition):
            for old_neighbor_id in G.neighbors(old_node_id):
                neighbor_id = partition[old_neighbor_id]
                if node_id != neighbor_id:
                    grouped_G.add_edge(node_id, neighbor_id)

        if weighted:
            grouped_G.graph['node_weight_attr'] = 'weight'

        return grouped_G


    def research(self) -> None:
        cr_list = [0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]

        pg_path = './data/physical_graphs/3_2x1.txt'
        PG = input_networkx_graph_from_file(pg_path)

        for input_dir in self.data_dirs:
            for graph_file in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, graph_file)): # and 'mk_test' in graph_file:
                    g_path = os.path.join(input_dir, graph_file)
                    print(g_path)
                    for cr in cr_list:
                        print(cr)
                        print('!!!!!!!', g_path, '!!!!!!!!!!!')

                        self.CUT_RATIO = cr

                        # weighted
                        G_weighted = input_networkx_graph_from_file(g_path)

                        (n, weighted_partition) = mk.mk_part(G_weighted, PG)

                        if weighted_partition is None:
                            print(g_path, self.CUT_RATIO)
                            continue
                        
                        mk_graph = mk.group_mk(G_weighted, weighted_partition)
                        mk.write_mk(graph_file.replace('.', '_weighted.'), mk_graph, weighted_partition)

                        # unweighted
                        G_unweighted = input_networkx_unweighted_graph_from_file(g_path)

                        (n, unweighted_partition) = mk.mk_part(G_unweighted, PG)
                        
                        if unweighted_partition is None:
                            print(g_path, self.CUT_RATIO)
                            continue

                        mk_graph = mk.group_mk(G_unweighted, unweighted_partition)

                        mk.write_mk(graph_file.replace('.', '_unweighted.'), mk_graph, unweighted_partition)


if __name__ == '__main__':
    # g_path = './data/testing_graphs/mk_test_tiny_ex.txt'
    # g_path = './data/testing_graphs/mk_test_mini_ex.txt'
    # g_path = './data/testing_graphs/mk_test_small_ex.txt'
    # g_path = './data/triangle/graphs/triadag10_3.txt'
    g_path = './data/sausages/dagD39.txt' # 65 вершин
    # g_path = './data/sausages/dagR62.txt' # 200 вершин

    G = input_networkx_graph_from_file(g_path)

    pg_path = './data/physical_graphs/3_2x1.txt'
    PG = input_networkx_graph_from_file(pg_path)

    graph_dirs = [
        './data/testing_graphs',
        './data/triangle/graphs',
        './data/sausages',
    ]

    mk = MK(data_dirs=graph_dirs)

    # mk.research()

    mk.CUT_RATIO = 0.07
    mk.CUT_RATIO = 0.4
    mk.CUT_RATIO = 0.45
    # mk.CUT_RATIO = 0.454
    # mk.CUT_RATIO = 0.47
    # mk.CUT_RATIO = 0.48
    # mk.CUT_RATIO = 0.5

    # G.remove_edges_from(list(G.edges))
    # nparts = 32

    # partition = mk.do_metis(G, nparts, 800)
    (n, partition) = mk.mk_part(G, PG)

    print(partition)
    print(set(partition))
    print('num of groups: ', len(set(partition)))
    print('partition CR: ', calc_cut_ratio(G, partition))

    weights = defaultdict(int)
    for i, group in enumerate(partition):
        weights[group] += G.nodes[i]['weight']
    print('weights: ', sorted(weights.values()))
    print(f'min weight: {min(weights.values())}, max weight: {max(weights.values())}')

    for i in sorted(list(set(partition))):
        print(partition.count(i), end=' ')
    print()

    mk_graph = mk.group_mk(G, partition)

    # print(G)
    # print(mk_graph)

    mk.CUT_RATIO = 0.4
    mk.write_mk(g_path.split('/')[-1].replace('.', '_weighted.'), mk_graph, partition)
