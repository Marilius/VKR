from os import listdir
from os.path import isfile, join

import json

from helpers import input_networkx_graph_from_file

path = './results2/MK_greed_greed_with_geq_cr/weighted'
print(1)


graphs = [
    'testing_graphs/16_envelope_mk_eq.time',
    'testing_graphs/16_envelope_mk_rand.time',
    'testing_graphs/64_envelope_mk_eq.time',
    'testing_graphs/64_envelope_mk_rand.time',
    'rand/dag26.time',
    'rand/dag15.time',
    'rand/dag16.time',
    'rand/dag13.time',
    'rand/dag0.time',
    'sausages/dagA15.time',
    'sausages/dagH28.time',
    'sausages/dagK43.time',
    'sausages/dagN19.time',
    'sausages/dagR49.time',
    'triangle/triadag10_5.time',
    'triangle/triadag15_4.time',
    'triangle/triadag20_5.time',
    'triangle/triadag25_0.time',
    'triangle/triadag30_7.time',
]

for graph in graphs:
    path = f'./data/{graph}'.replace('.time', '.txt').replace('triangle', 'triangle/graphs')
    g = input_networkx_graph_from_file(path)
    print(f'\item ${g.graph['graph_name']}$: {len(g.nodes())} вершин, {len(g.edges())} ребер;')


# for folder in listdir(path):
#     for file in listdir(join(path, folder)):
#         if isfile(join(path, folder, file)):
#             with open(join(path, folder, file), 'r') as f:
#                 lines = f.readlines()
#                 # print(line)
#                 if len(lines) >= 375:
#                     print(join(path, folder, file), end=' ')
                
#                     line = lines[-1]
#                     # print(line)
#                     _, pg, _, cr, cr_max, f_val, partition = line.split(maxsplit=6)
#                     # print(partition)
#                     partition = json.loads(partition.strip())
#                     print(len(partition))

        # print(file)