from .main import GAP

graph_dirs = [
    (r'./data/sausages', './results/GAP/sausages'),
    (r'./data/triangle/graphs', './results/GAP/triangle'),
    (r'./data/rand', './results/GAP/rand'),
    (r'./data/testing_graphs', './results/GAP/testing_graphs'),
]
gap_alg = GAP(article=True, graph_dirs=graph_dirs)
# необходимо чтобы граф был связный
# достаточно(для запуска алгоритма), чтобы было верно следующее:
# (minimum_cut/2)*R0 >= кол-во процессоров
# minimum_cut/2 (точнее количество вершин, которые принадлежат этим рёбрам)
# ещё лучше, если значение выражения больше желаемого размера популяции

# graph_name = r'../data/sausages/dagA0.txt'
# physical_graph_name = r'../data/physical_graphs/1234x3.txt'
# G = input_graph_from_file(graph_name)
# PG = input_graph_from_file(physical_graph_name)    

# physical_graph_dirs = [
#     r'../data/physical_graphs',
# ]

gap_alg.research()