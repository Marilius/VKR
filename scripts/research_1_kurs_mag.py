from algo.helpers import input_graph, fix_rand_graph_file

from algo.genetic import GeneticPartitioner

import networkx as nx

from os.path import isfile, join
from copy import deepcopy

params: list[dict] = []

genetic_partitioner: GeneticPartitioner = GeneticPartitioner()

graphs2part: list[tuple[str, str, str]] = [
    (r'./data/gen_data', r'./results/greed/{}gen_data', r'5_4_3_2_4000_10_100_4.0_0.6_True.graph'),
    (r'./data/gen_data', r'./results/greed/{}gen_data', r'4_1_4000_10_100_4.0_0.4_True.graph'),

    (r'./data/random', r'./results/greed/{}random', r'dag7.txt'),
    (r'./data/random', r'./results/greed/{}random', r'dag30.txt'),


    (r'./data/layered', r'./results/greed/{}layered', r'dagA15.txt'),
    (r'./data/layered', r'./results/greed/{}layered', r'dagR72.txt'),

    (r'./data/triangle', r'./results/greed/{}triangle', r'triadag60_9.txt'),
    (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_0.txt'),
]

physical_graphs: list[str] = [
    r'./data/processor_sets/5_4_3_2x1.txt',
    r'./data/processor_sets/4_1x1.txt',
]

cr_list: list[float] = [0.4, 0.6, 1]
ga_params: list[dict] = [
    {
        'do_transit' : False,
        'population_size': 300,
        'ngen_without_1pct_improvement': 10,
        'mate': {
            'name': 'cxTwoPoint',
        },
        'mutate': {
            'name': 'mutFlipBit',
            'indpb': 0.05,
        },
        'select': {
            'name': 'selTournament',
            'tournsize': 3,
        },
    },
    
    {
        'do_transit' : False,
        'population_size': 300,
        'ngen_without_1pct_improvement': 10,
        'mate': {
            'name': 'cxTwoPoint',
        },
        'mutate': {
            'name': 'mutShuffleIndexes',
            'indpb': 0.05,
        },
        'select': {
            'name': 'selTournament',
            'tournsize': 3,
        },
    },
    
    {
        'do_transit' : False,
        'population_size': 300,
        'ngen_without_1pct_improvement': 10,
        'mate': {
            'name': 'cxTwoPoint',
        },
        'mutate': {
            'name': 'mutShuffleIndexes',
            'indpb': 0.05,
        },
        'select': {
            'name': 'selTournament',
            'tournsize': 3,
        },
    },
]

for input_dir, output_dir, graph_file in graphs2part:
    graph_path: str = join(input_dir, graph_file)

    if isfile(graph_path) and 'partition' not in graph_file:
        weighted_graph: nx.Graph = input_graph(graph_path)

        if list(sorted(list(weighted_graph.nodes))) != list(range(len(weighted_graph.nodes))):
            print('something is wrong with the graph', graph_path)
            print('fixing graph...')
            fix_rand_graph_file(graph_path)
            weighted_graph: nx.Graph = input_graph(graph_path)

        for physical_graph_path in physical_graphs:
            if 'gen_data' in input_dir:
                try:
                    print(physical_graph_path)
                    pg = physical_graph_path.rsplit('/', 1)[0].removesuffix('.txt').split('x')
                    pg_prefix = (pg[0] + '_') * int(pg[1])

                    L, min_l, max_l, N, cr_gen, shuffle = graph_file.removesuffix('.graph').removeprefix(pg_prefix).split('_')

                    L, min_l, max_l, N, cr_gen = int(L), int(min_l), int(max_l), float(N), float(cr_gen)
                except Exception as e:
                    continue

            for cr in cr_list:
                for ga_param in ga_params:
                    ga = deepcopy(ga_param)
                    ga['seed'] = abs(hash(f'{graph_file} {physical_graph_path} {cr}')) % (10 ** 8)
                    
                    params.append(
                        {
                            'input_dir': input_dir,
                            'output_dir': output_dir,
                            'graph_file': graph_file,
                            'physical_graph_dir': physical_graph_path.rsplit('/', 1)[0],
                            'physical_graph_path': physical_graph_path.rsplit('/', 1)[1],
                            'ga_params': ga,
                        }
                    )

for param in params:
    for _ in range(10):
        param0 = deepcopy(param)
        genetic_partitioner.do_genetic(**deepcopy(param0))
        param0['ga_params']['do_transit'] = True
        genetic_partitioner.do_genetic(**deepcopy(param0))
