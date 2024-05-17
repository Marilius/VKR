from os import listdir
from os.path import isfile, join

import json

from helpers import input_networkx_graph_from_file

path = './results2/MK_greed_greed_with_geq_cr/weighted'
print(1)


for folder in listdir(path):
    for file in listdir(join(path, folder)):
        if isfile(join(path, folder, file)):
            with open(join(path, folder, file), 'r') as f:
                lines = f.readlines()
                # print(line)
                if len(lines) >= 375:
                    print(join(path, folder, file), end=' ')
                
                    line = lines[-1]
                    # print(line)
                    _, pg, _, cr, cr_max, f_val, partition = line.split(maxsplit=6)
                    # print(partition)
                    partition = json.loads(partition.strip())
                    print(len(partition))

        # print(file)