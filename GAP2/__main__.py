from .main import GAP2


print('lol')

graph_dirs = [
    (r'./data/testing_graphs', './results/GAP2/testing_graphs'),
    (r'./data/triangle/graphs', './results/GAP2/triangle'),
    (r'./data/sausages', './results/GAP2/sausages'),
    (r'./data/rand', './results/GAP2/rand'),
]

gap_algo = GAP2(article=True, graph_dirs=graph_dirs, iter_max=100)
# graph_path = r'./data/triangle/graphs/triadag10_0.txt'

# physical_graph_path = r'./data/physical_graphs/1234x3.txt'
# G = input_graph_from_file()
# PG = input_graph_from_file(r'../data/test_gp/0.txt')

# GAP2.do_gap(graph_path, physical_graph_path)


gap_algo.research()
