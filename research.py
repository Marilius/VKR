from helpers import input_graph

from mk.main import MKPartitioner
from greed.main import GreedPartitioner

import networkx as nx

from os import listdir
from os.path import isfile, join

import time


greed_partitioner: GreedPartitioner = GreedPartitioner()
mk_partitioner: MKPartitioner = MKPartitioner()

graph_dirs = [
    # (r'./data/triangle/graphs', r'./results/greed/{}triangle'),
    # (r'./data/rand', r'./results/greed/{}rand'),
    # (r'./data/testing_graphs', r'./results/greed/{}testing_graphs'),
    # (r'./data/sausages', r'./results/greed/{}sausages'),
    (r'./data/gen_data', r'./results/greed/{}gen_data'),
]

physical_graph_dirs = [
    r'./data/physical_graphs',
]

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
    # 'dagA15.txt',
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

graphs = [graph for graph in listdir('./data/gen_data') if 'partition' not in graph]

# cr_list_little = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
cr_list_little = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
cr_list_big = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
cr_list = cr_list_little + cr_list_big
# cr_list =  cr_list_big
cr_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]
# cr_list = [0.1, 0.15]

for input_dir, output_dir in graph_dirs:
    for graph_file in listdir(input_dir):
        if isfile(join(input_dir, graph_file)) and graph_file in graphs and 'partition' not in graph_file:
            for physical_graph_dir in physical_graph_dirs:
                for physical_graph_path in listdir(physical_graph_dir):
                    if isfile(join(physical_graph_dir, physical_graph_path)): # and '1_4x2.txt' in physical_graph_path:
                        graph_name = physical_graph_path.split('.')[0]
                        graph_name = graph_name.split('x')
                        # graph_name = (graph_name[0] + '_') * int(graph_name[1])
                        # graph_name = graph_name.strip('_')
                        
                        # if '1_4_1_4_4000_10_100_2.0_0.1_True.graph' != graph_file:
                            # continue
                        # if '1_4x2' not in physical_graph_path:
                            # continue
                        # if '1_4x2' not in physical_graph_path or '4000' not in graph_file or '10_100' not in graph_file or '2.0' not in graph_file or '0.1' not in graph_file:
                            # continue
                        # if 'True' not in graph_file:
                            # continue
                        
                        if '5_4_3_2_5_4_3_2_2000_10_100_1.5_0.1_True.graph' != graph_file:
                            continue    
                        
                        if 'gen_data' in input_dir and graph_file.count(graph_name[0]) != int(graph_name[1]):
                            continue

                        for cr in cr_list:
                            weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))

                            # if len(weighted_graph.nodes) > 100:
                                # continue

                            if list(sorted(list(weighted_graph.nodes))) != list(range(len(weighted_graph.nodes))):
                                continue

                            # unweighted_graph = input_networkx_unweighted_graph_from_file(join(input_dir, graph_file))
                            physical_graph = input_graph(join(physical_graph_dir, physical_graph_path))
                            # weighted
                            start_time = time.time()
                            initial_weighted_partition = greed_partitioner.do_metis_with_pg(weighted_graph, physical_graph, cr_max=cr)
                            greed_partitioner.write_results(join(output_dir.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph, cr, start_time)
                            start_time = time.time()
                            weighted_partition = greed_partitioner.do_greed(weighted_graph, physical_graph, initial_weighted_partition, cr)
                            greed_partitioner.write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, cr, start_time)

                            # # unweighted
                            # start_time = time.time()
                            # initial_unweighted_partition = self.do_metis_with_pg(unweighted_graph, physical_graph)
                            # self.write_results(join(output_dir.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph, start_time)
                            # start_time = time.time()
                            # unweighted_partition = self.do_greed(weighted_graph, physical_graph, initial_unweighted_partition)
                            # self.write_results(join(output_dir.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph, start_time)

                            # print('from mk weighted')
                            # self.CUT_RATIO = 1 
                            # output_dir_mk = output_dir.replace('greed', 'greed_from_mk_weighted')
                            # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.', str(cr) + '.')
                            # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_weighted.').replace('.txt', str(cr) + '.' + 'mapping')
                            # mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                            # mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                            # physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                            # initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, check_cache=True)
                            # initial_unweighted_partition = self.do_metis_with_pg(mk_graph_unweighted, physical_graph, check_cache=True)
                            # weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                            # unweighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                            # initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                            # initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                            # weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                            # unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                            # self.CUT_RATIO = cr
                            # self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                            # self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                            # print('unweighted')
                            # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                            # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)

                            # print('from mk UNweighted')
                            # self.CUT_RATIO = 1
                            # output_dir_mk = output_dir.replace('greed', 'greed_from_mk_unweighted')
                            # mk_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.', str(cr) + '.')
                            # mk_data_path = self.MK_DIR + '/' + graph_file.replace('.', '_unweighted.').replace('.txt', str(cr) + '.' + 'mapping')
                            # mk_graph_weighted = input_networkx_graph_from_file(mk_path)
                            # mk_graph_unweighted = input_networkx_unweighted_graph_from_file(mk_path)
                            # physical_graph = input_networkx_graph_from_file(join(physical_graph_dir, physical_graph_path))

                            # initial_weighted_partition = self.do_metis_with_pg(mk_graph_weighted, physical_graph, check_cache=True)
                            # initial_unweighted_partition = self.do_metis_with_pg(mk_graph_unweighted, physical_graph, check_cache=True)
                            # weighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_weighted_partition)
                            # unweighted_partition = self.do_greed(mk_graph_weighted, physical_graph, initial_unweighted_partition)

                            # initial_weighted_partition = do_unpack_mk(initial_weighted_partition, mk_data_path)
                            # initial_unweighted_partition = do_unpack_mk(initial_unweighted_partition, mk_data_path)
                            # weighted_partition = do_unpack_mk(weighted_partition, mk_data_path)
                            # unweighted_partition = do_unpack_mk(unweighted_partition, mk_data_path)

                            # self.CUT_RATIO = cr
                            # self.write_results(join(output_dir_mk.format('weighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_weighted_partition, weighted_graph, physical_graph)
                            # self.write_results(join(output_dir_mk.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph)

                            # unweighted
                            # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file).replace('greed', 'metis'), join(physical_graph_dir, physical_graph_path), initial_unweighted_partition, weighted_graph, physical_graph)
                            # self.write_results(join(output_dir_mk.format('unweighted/'), graph_file), join(physical_graph_dir, physical_graph_path), unweighted_partition, weighted_graph, physical_graph)

                            # self.do_weighted_mk_with_geq_cr(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, cr)
                            
                            mk_partitioner.do_MK_greed_greed(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, cr, check_cache=True, steps_back=6)
                            
                            # self.do_MK_greed_greed_with_geq_cr(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path, check_cache=True, steps_back=6)

                            # self.do_simple_part(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path)
                            # self.write_metis_with_pg(input_dir, output_dir, graph_file, physical_graph_dir, physical_graph_path)