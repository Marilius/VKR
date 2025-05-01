from algo.helpers import input_graph, calc_cut_ratio, check_cut_ratio

from algo.helpers import f_new as f

from deap import creator, base, tools, algorithms
import networkx as nx

import time
import random

from os import makedirs
from os.path import join


class GeneticPartitioner:
    def write_results(self, path: str, physical_graph_path: str, partition: list[int], G: nx.Graph, PG: nx.Graph, cr_max: float, start_time: float) -> None:
        end_time = time.time()

        line2write = [
            path.split('/')[-1],  # The graph name
            physical_graph_path.split('/')[-1],  # The physical graph name
            calc_cut_ratio(G=G, partition=partition),  # The cut ratio
            cr_max,  # The maximum allowed cut ratio
            f(G, PG, partition, cr_max),  # The function value
            partition if check_cut_ratio(G, partition, cr_max) else None,  # The partition list
            '\n',  # A newline at the end of the line
        ]

        assert partition is None or len(set(partition)) <= len(PG.nodes)
        assert partition is None or len(partition) == len(G.nodes)

        makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        with open(path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))

        overall_time = end_time - start_time

        line2write = [
            path.split('/')[-1],  # The graph name
            physical_graph_path.split('/')[-1],  # The physical graph name
            cr_max,  # The maximum allowed cut ratio
            len(partition) if partition is not None else None,  # The number of partitions
            start_time,  # The start time
            str(overall_time),  # The overall time
            '\n',  # A newline at the end of the line
        ]

        time_path = path.replace('.txt', '.time').replace('.graph', '.time')
        with open(time_path, 'a+') as file:
            file.write(' '.join(map(str, line2write)))
            
    def genetic(self, G: nx.Graph, PG: nx.Graph, cr_max: float, check_cache: bool, seed: int | None) -> list[int]:
        def evalOneMax(individual):
            print(individual)
            print(f(G, PG, individual, cr_max))
            return [-f(G, PG, individual, cr_max)]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_int", random.randint, 0, len(PG) - 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(G))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=300)

        # NGEN=40
        # NGEN_WITHOUT_IMPROVEMENT=5
        # for gen in range(NGEN):
        f_best = tools.selBest(population, k=1)[0]
        ngen_without_improvement = 5
        while ngen_without_improvement:
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                print(ind.fitness.values, fit)
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            top1 = tools.selBest(population, k=1)[0]
            if top1.fitness > f_best.fitness:
                f_best = top1
                ngen_without_improvement = 5
            else:
                ngen_without_improvement -= 1
        print(f_best)
        return f_best
    
    def do_genetic(
        self,
        input_dir: str,
        output_dir: str,
        graph_file: str,
        physical_graph_dir: str,
        physical_graph_path: str,
        cr_max: float,
        check_cache: bool,
        seed: int | None,
    ) -> None:
        weighted_graph: nx.Graph = input_graph(join(input_dir, graph_file))
        physical_graph: nx.Graph = input_graph(join(physical_graph_dir, physical_graph_path))

        output_dir = output_dir.replace('greed', 'genetic')

        start_time = time.time()
        weighted_partition = self.genetic(weighted_graph, physical_graph, cr_max, check_cache, seed)
        print(join(output_dir.format('weighted/'), graph_file))
        print(join(physical_graph_dir, physical_graph_path))
        print('cr_max:', cr_max)
        print(f(weighted_graph, physical_graph, weighted_partition, cr_max), calc_cut_ratio(weighted_graph, weighted_partition))
        print('---')
        
        self.write_results(join(output_dir.format('weighted/'), graph_file), join(physical_graph_dir, physical_graph_path), weighted_partition, weighted_graph, physical_graph, cr_max, start_time)
