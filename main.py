# gen test
from helpers import input_generated_graph_and_processors_from_file
import metis

# path = './data/gen_data/1_1_1_1_300_5_25_500_200.txt'
# path = './data/gen_data/3_3_3_3_3000_5_25_500_200.txt'
path = './data/gen_data/3_3_3_3_300_5_25_500_200.txt'

G, processors, params, exact_partition = input_generated_graph_and_processors_from_file(path)
nparts = len(processors)
ufactor = 1
recursive = True

print(len(G.nodes()), len(G.edges()))

(edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)
print(edgecuts)

partition = [0] * len(G.nodes)
for new_i, i in enumerate(list(G.nodes)):
    partition[i] = partition2parse[new_i]

for new_i, i in enumerate(sorted(list(set(partition)))):
    for j in range(len(partition)):
        if partition[j] == i:
            partition[j] = new_i

weights = [0] * nparts
for i in range(len(partition)):
    weights[partition[i]] += G.nodes[i]['weight']
print(weights)
print(partition)


print(exact_partition)
print(len(exact_partition), len(partition))
weights = [0] * nparts
for i in range(len(exact_partition)):
    weights[exact_partition[i]] += G.nodes[i]['weight']
print(weights)
