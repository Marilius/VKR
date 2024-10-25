# # gen test
from greed import Greed
from helpers import input_generated_graph_and_processors_from_file, input_generated_graph_partition, input_graph
import metis

# # path = './data/gen_data/1_1_1_1_300_5_25_500_200.txt'
# # path = './data/gen_data/3_3_3_3_3000_5_25_500_200.txt'
# # path = './data/gen_data/1_2_3_4_300_10_20_8_0.3_True.graph'
# # path = './data/gen_data/2_2_2_2_300_10_20_12_0.3_True.graph'
# # path = './data/gen_data/2_2_2_2_500_10_20_12_0.3_True.graph'
# path = './data/gen_data/2_2_2_2_500_10_20_20.0_0.3_True.graph'
path = './data/gen_data/4_1_4_1_4000_10_100_2.0_0.1_True.graph'

graph, processors, params = input_generated_graph_and_processors_from_file(path)
exact_partition = input_generated_graph_partition(path.replace('graph', 'partition'))

pg_path = './data/physical_graphs/1_4x2.txt'
physical_graph = input_graph(pg_path)

partitioner = Greed()

nparts = len(processors)
ufactor = 1000
recursive = True

# print(len(G.nodes()), len(G.edges()))

# (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)
# print(edgecuts)

partition = partitioner.run(graph, physical_graph, cr=0.1)

weights = [0] * nparts
for i in range(len(partition)):
    weights[partition[i]] += graph.nodes[i]['weight']
print(weights)
# print(partition)

partition = partitioner.run(graph, physical_graph, cr=0.15)

weights = [0] * nparts
for i in range(len(partition)):
    weights[partition[i]] += graph.nodes[i]['weight']
print(weights)
# print(partition)

# print(exact_partition)
# print(len(exact_partition), len(partition))
# weights = [0] * nparts
# for i in range(len(exact_partition)):
#     weights[exact_partition[i]] += G.nodes[i]['weight']
# print(weights)



# path = path.replace('True', 'False')
# G, processors, params = input_generated_graph_and_processors_from_file(path)
# exact_partition = input_generated_graph_partition(path.replace('graph', 'partition'))

# print('nodes_cnt =', len(G.nodes()), 'edges_cnt =', len(G.edges()))

# (edgecuts, partition2parse) = metis.part_graph(G, nparts, objtype='cut', ncuts=10, ufactor=ufactor, recursive=recursive)
# print('edgecuts = ', edgecuts)

# partition = [0] * len(G.nodes)
# for new_i, i in enumerate(list(G.nodes)):
#     partition[i] = partition2parse[new_i]

# for new_i, i in enumerate(sorted(list(set(partition)))):
#     for j in range(len(partition)):
#         if partition[j] == i:
#             partition[j] = new_i

# weights = [0] * nparts
# for i in range(len(partition)):
#     weights[partition[i]] += G.nodes[i]['weight']
# print(weights)
# print(partition)


# print(exact_partition)
# print(len(exact_partition), len(partition))
# weights = [0] * nparts
# for i in range(len(exact_partition)):
#     weights[exact_partition[i]] += G.nodes[i]['weight']
# print(weights)