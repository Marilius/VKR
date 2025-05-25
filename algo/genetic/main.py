from typing import Callable
from algo.helpers import input_graph, calc_cut_ratio
from algo.helpers import f_new as f

from algo.helpers import create_transit_graph, unpack_transit_partition

from algo.base_partitioner import BasePartitioner

from deap import creator, base, tools, algorithms
import networkx as nx

import time
import random

from os import makedirs
from os.path import join

import json
from copy import deepcopy


def step_first(n):
    if n < 100:
        d = 5
    elif n < 300:
        d = 10
    else:
        d = 20
    return d

def step_second(n):
    if n < 100:
        d = 10
    elif n < 300:
        d = 15
    else:
        d = 30
    return d

def step_third(n):
    if n < 100:
        d = 15
    elif n < 300:
        d = 20
    else:
        d = 40
    return d

def sqrt_d_func(n):
    return round(n**0.5)

d_functions: dict[str, Callable] = {
    'step_first': step_first,
    'step_second': step_second,
    'step_third': step_third,
    'sqrt': sqrt_d_func,
}


class GeneticPartitioner:
    def write_results(self, path: str, physical_graph_path: str, partition: list[int], G: nx.DiGraph, PG: nx.Graph, start_time: float, ga_params: dict = {}) -> None:
        end_time = time.time()

        assert partition is None or len(set(partition)) <= len(PG.nodes)
        assert partition is None or len(partition) == len(G.nodes)

        d = {
            'graph_name': path.split('/')[-1],
            'physical_graph_name': physical_graph_path.split('/')[-1],
            'ga_params': ga_params,
            'f_val': f(G, PG, partition),
            'partition': partition,
        }
        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        with open(path, 'a+') as file:
            file.write(json.dumps(d))
            file.write('\n')

        overall_time = end_time - start_time
        d = {
            'graph_name': path.split('/')[-1],
            'physical_graph_name': physical_graph_path.split('/')[-1],
            'ga_params': ga_params,
            'start_time': start_time,
            'overall_time': overall_time,
        }
        time_path = path.replace('.txt', '.time').replace('.graph', '.time')
        with open(time_path, 'a+') as file:
            file.write(json.dumps(d))
            file.write('\n')

    def genetic(self, G: nx.DiGraph, PG: nx.Graph, ga_params: dict) -> list[int]:
        do_transit: bool = ga_params.get('do_transit', False)

        # default values
        subgraphs_data: dict[int, tuple[set[int], int, set[int]]] | None = None
        node_to_subgraph: dict[int, int] | None = None
        individual_len: int = len(G)

        if do_transit:
            seed = ga_params.get('seed')
            base_partitioner: BasePartitioner = BasePartitioner()
            
            # n = len(G)
            # if n < 50:
            #     d = 5
            # elif n < 100:
            #     d = 10
            # elif n < 400:
            #     d = 20
            # else:
            #     d = 50
                

            # else:
            #     d = 20
            d_func: str = ga_params.get('d_func')
            d = d_functions[d_func](len(G))
            
            undirected_graph: nx.Graph = G.to_undirected()
            
            print('AAAAAAAAAAAAAAAA', G, len(G) // d, 1000, False, seed)
            partition: list[int] = base_partitioner.metis_part(undirected_graph, len(G) // d, 100, False, seed)

            transit_graph, subgraphs_data, node_to_subgraph, _ = create_transit_graph(G, partition)
            individual_len = len(subgraphs_data)

        def evalOneMax(individual: list[int]):
            if do_transit:
                return [f(transit_graph, PG, individual, subgraphs_data, node_to_subgraph)]
            else:
                return [f(G, PG, individual)]

        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register('attr_int', random.randint, 0, len(PG) - 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_int, n=individual_len)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        mate: dict = ga_params.get('mate')
        mate_name: str = mate.pop('name')
        mutate: dict = ga_params.get('mutate')
        mutate_name: str = mutate.pop('name')
        select: dict = ga_params.get('select')
        select_name: str = select.pop('name')

        toolbox.register('evaluate', evalOneMax)
        toolbox.register('mate', getattr(tools, mate_name), **mate)
        toolbox.register('mutate', getattr(tools, mutate_name), **mutate)
        toolbox.register('select', getattr(tools, select_name), **select)

        population_size = ga_params.get('population_size')
        population = toolbox.population(n=population_size)


        ngen_without_1pct_improvement_base: int = ga_params.get('ngen_without_1pct_improvement')
        ngen_without_1pct_improvement = ngen_without_1pct_improvement_base

        f_best = tools.selBest(population, k=1)[0]
        while ngen_without_1pct_improvement:
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                print(ind.fitness.values, fit)
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            top1 = tools.selBest(population, k=1)[0]
            if top1.fitness > f_best.fitness:
                print('top1.fitness', top1.fitness.values[0])
                print('f_best.fitness.values[0] / top1.fitness.values[0]', f_best.fitness, top1.fitness)
                if f_best.fitness.values and f_best.fitness.values[0] / top1.fitness.values[0] >= 1.01:
                    ngen_without_1pct_improvement = ngen_without_1pct_improvement_base
                f_best = top1
            else:
                ngen_without_1pct_improvement -= 1

        print(f_best)


        if do_transit:
            f1 = f(transit_graph, PG, f_best, subgraphs_data, node_to_subgraph)
            f_best = unpack_transit_partition(f_best, subgraphs_data)
            f2 = f(G, PG, f_best)
            
            assert abs(f1 - f2) < 0.00001, f'{f1} {f2}'

        return f_best
    
    def do_genetic(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        ga_params: dict,
    ) -> None:
        weighted_graph: nx.DiGraph = input_graph(join(input_dir, graph_file))
        physical_graph: nx.DiGraph = input_graph(join(physical_graph_dir, physical_graph_path))
        
        ga_params_copy = deepcopy(ga_params)

        output_dir = output_dir.replace('greed', 'genetic')

        start_time = time.time()
        weighted_partition = self.genetic(weighted_graph, physical_graph, ga_params)
        print(join(output_dir.format('weighted/'), graph_file))
        print(join(physical_graph_dir, physical_graph_path))
        print(f(weighted_graph, physical_graph, weighted_partition), calc_cut_ratio(weighted_graph, weighted_partition))
        print('---')
        
        self.write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, start_time, ga_params_copy)
