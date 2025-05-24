from algo.helpers import input_graph, fix_rand_graph_file

from algo.genetic import GeneticPartitioner

import networkx as nx

from os.path import isfile, join
from os import listdir
from copy import deepcopy
import random

params: list[dict] = []

genetic_partitioner: GeneticPartitioner = GeneticPartitioner()

graphs2part_random: list[tuple[str, str, str]] = [(r'./data/random', r'./results/greed/{}random', graph) for graph in listdir(r'./data/random')]
graphs2part_layered: list[tuple[str, str, str]] = [(r'./data/layered', r'./results/greed/{}layered', graph) for graph in listdir(r'./data/layered')]

triangle = sorted(listdir(r'./data/triangle'))
triangle = triangle[:triangle.index('triadag35_9.txt')+1] 
graphs2part_triangle: list[tuple[str, str, str]] = [(r'./data/triangle', r'./results/greed/{}triangle', graph) for graph in triangle]
# graphs2part_triangle: list[tuple[str, str, str]] = [(r'./data/triangle', r'./results/greed/{}triangle', graph) for graph in sorted(listdir(r'./data/triangle'))]

graphs2part: list[tuple[str, str, str]] = [
    # *graphs2part_random[3:],
    # (r'./data/random', r'./results/greed/{}random', r'dag7.txt'),
    # (r'./data/random', r'./results/greed/{}random', r'dag30.txt'),

    # *graphs2part_layered,
    # (r'./data/layered', r'./results/greed/{}layered', r'dagA15.txt'),
    # (r'./data/layered', r'./results/greed/{}layered', r'dagR72.txt'),

    graphs2part_triangle[0],
    graphs2part_triangle[10],
    graphs2part_triangle[20],
    graphs2part_triangle[30],
    graphs2part_triangle[40],
    graphs2part_triangle[50],
    # *graphs2part_triangle[-10:],
    # *graphs2part_triangle[-10:],
    # *graphs2part_triangle[-10:],
    # *graphs2part_triangle[-10:],
    # *graphs2part_triangle[:20],
    # *graphs2part_triangle[20:30],
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_0.txt'),
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_1.txt'),
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_2.txt'),
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_3.txt'),
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_4.txt'),
    
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag60_9.txt'),
    # (r'./data/triangle', r'./results/greed/{}triangle', r'triadag30_0.txt'),
]

physical_graphs: list[str] = [
    r'./data/processor_sets/5_4_3_2x1.txt',
    r'./data/processor_sets/4_1x1.txt',
]

ga_params: list[dict] = [
    {
        'do_transit' : False,
        'population_size': 300,
        'ngen_without_1pct_improvement': 15,
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
        'ngen_without_1pct_improvement': 15,
        'mate': {
            'name': 'cxTwoPoint',
        },
        'mutate': {
            'name': 'mutFlipBit',
            'indpb': 0.1,
        },
        'select': {
            'name': 'selTournament',
            'tournsize': 3,
        },
    },
    
    {
        'do_transit' : False,
        'population_size': 300,
        'ngen_without_1pct_improvement': 15,
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

d_functions = [
    'step_first', 'step_second', 'step_third', 'sqrt'
]

if __name__ == '__main__':
    print('RUNNING...')
    
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
                for ga_param in ga_params:
                    ga = deepcopy(ga_param)
                    ga['do_transit'] = False
                    
                    ga['seed'] = random.randint(0, 10 ** 8)
                    print(ga['seed'])
                    
                    # no transit
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
                    
                    # transit
                    for d_func in d_functions:
                        ga = deepcopy(ga_param)
                        ga['do_transit'] = True
                        ga['d_func'] = d_func
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

    print(params)

    for param in params:
        for _ in range(10):
            param0 = deepcopy(param)
            param0['ga_params']['seed'] = random.randint(0, 10 ** 8)
            genetic_partitioner.do_genetic(**deepcopy(param0))
            # param0['ga_params']['do_transit'] = True
            # genetic_partitioner.do_genetic(**deepcopy(param0))
