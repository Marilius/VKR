from os import listdir
from os.path import isfile, join

import json

from helpers import input_networkx_graph_from_file

# path = './results2/MK_greed_greed_with_geq_cr/weighted'
path_rand = './data/rand'
path_sausages = './data/sausages'
path_testing_graphs = './data/testing_graphs'
path_triangle = './data/triangle/graphs'
print(1)

path = path_triangle
n = 350
delta = 100

for file in sorted(listdir(path)):
    if isfile(join(path, file)):
        with open(join(path, file), 'r') as f:
            # lines = f.readlines()
            # print(line)
            # if len(lines) == n + 1:
            graph = input_networkx_graph_from_file(join(path, file))
            if n - delta <= len(graph.nodes) <= n + delta:
                print(join(path, file), len(graph.nodes))
            
                # line = lines[-1]
                # print(line)
                # _, pg, _, cr, cr_max, f_val, partition = line.split(maxsplit=6)
                # print(partition)
                # partition = json.loads(partition.strip())
                # print(len(partition))

        # print(file)