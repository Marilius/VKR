from algo.helpers import input_graph, fix_rand_graph_file

from algo.mk.main import MKPartitioner
from algo.greed.main import GreedPartitioner

import networkx as nx

from os import listdir
from os.path import isfile, join

from joblib import Parallel, delayed


params: list[dict] = []

greed_partitioner: GreedPartitioner = GreedPartitioner()
mk_partitioner: MKPartitioner = MKPartitioner()

# graph classes to be run and corresponding output directories
graph_dirs = [
    # (r'./data/triangle/graphs', r'./results/greed/{}triangle'),
    # (r'./data/rand', r'./results/greed/{}rand'),
    # (r'./data/testing_graphs', r'./results/greed/{}testing_graphs'),
    # (r'./data/sausages', r'./results/greed/{}sausages'),
    # (r'./data/gen_data', r'./results/greed/{}gen_data'),
    
    # (r'./data/layered_all', r'./results/greed/{}layered_all'),
    # (r'./data/rand_first100', r'./results/greed/{}rand_first100'),
    (r'./data/triangle_new', r'./results/greed/{}triangle_new'),
]

physical_graph_dirs = [
    r'./data/physical_graphs',
]

cr_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]

for input_dir, output_dir in graph_dirs:
    graph_names = sorted(listdir(input_dir), key=lambda x: len(input_graph(join(input_dir, x)).nodes))
    
    for graph_file in graph_names:
        graph_path: str = join(input_dir, graph_file)
        
        if 'triadag35' in graph_file or 'triadag4' in graph_file or 'triadag5' in graph_file or 'triadag6' in graph_file:
            pass
        else:
            continue
        
        if isfile(graph_path) and 'partition' not in graph_file:
            weighted_graph: nx.Graph = input_graph(graph_path)
            
            if list(sorted(list(weighted_graph.nodes))) != list(range(len(weighted_graph.nodes))):
                print('something is wrong with the graph', graph_path)
                print('fixing graph...')
                fix_rand_graph_file(graph_path)
                weighted_graph: nx.Graph = input_graph(graph_path)

            for physical_graph_dir in physical_graph_dirs:
                for physical_graph_path in listdir(physical_graph_dir):
                    if isfile(join(physical_graph_dir, physical_graph_path)):
                        if 'gen_data' in input_dir:
                            try:
                                pg = physical_graph_path.removesuffix('.txt').split('x')
                                pg_prefix = (pg[0] + '_') * int(pg[1])
                                
                                L, min_l, max_l, N, cr_gen, shuffle = graph_file.removesuffix('.graph').removeprefix(pg_prefix).split('_')
                                
                                L, min_l, max_l, N, cr_gen = int(L), int(min_l), int(max_l), float(N), float(cr_gen)
                            except Exception as e:
                                continue

                        for cr in cr_list:
                            params.append(
                                {
                                    'input_dir': input_dir,
                                    'output_dir': output_dir,
                                    'graph_file': graph_file,
                                    'physical_graph_dir': physical_graph_dir,
                                    'physical_graph_path': physical_graph_path,
                                    'cr_max': cr, 
                                    'check_cache': False, 
                                    'steps_back': 6,
                                    'seed': abs(hash(f'{graph_file} {physical_graph_path} {cr}')) % (10 ** 8),
                                }
                            )

# run simple part
# Parallel(n_jobs=-1)(delayed(greed_partitioner.do_simple_part)(**param) for param in [{key: value for key, value in d.items() if key not in ['seed', 'check_cache', 'steps_back']} for d in params])

# run greed algo
# Parallel(n_jobs=-1)(delayed(greed_partitioner.run_from_paths)(**param) for param in [{key: value for key, value in d.items() if key not in ['steps_back']} for d in params])

# run mk algo
# Parallel(n_jobs=-1)(delayed(mk_partitioner.do_MK_greed_greed)(**param) for param in params)

# run mk algo with greater cr
# Parallel(n_jobs=10)(delayed(mk_partitioner.do_MK_greed_greed_with_geq_cr)(**param) for param in params)
