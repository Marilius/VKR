import networkx as nx
import metis


# For k-way clustering, the appropriate options are:

# objtype   = 'cut' or 'vol'
# ctype     = 'rm' or 'shem'
# iptype    = 'grow', 'random', 'edge', 'node'
# rtype     = 'fm', 'greedy', 'sep2sided', 'sep1sided'
# ncuts     = integer, number of cut attempts (default = 1)
# niter     = integer, number of iterations (default = 10)
# ufactor   = integer, maximum load imbalance of (1+x)/1000
# minconn   = bool, minimize degree of subdomain graph
# contig    = bool, force contiguous partitions
# seed      = integer, RNG seed
# numbering = 0 (C-style) or 1 (Fortran-style) indices
# dbglvl    = Debug flag bitfield


G = metis.example_networkx()
(edgecuts, parts) = metis.part_graph(G, 3)
colors = ['red','blue','green']
for i, p in enumerate(parts):
    G.node[i]['color'] = colors[p]

