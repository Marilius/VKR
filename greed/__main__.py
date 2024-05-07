from .main import Greed

from helpers import input_networkx_graph_from_file, input_networkx_unweighted_graph_from_file

import sys


match sys.argv[1:]:
    case ['--G', g_path, '--PG', pg_path, '--CR', cr, '--OUTPUT_DIR', output_dir]:
        greed = Greed()
        CUT_RATIO = cr

        weighted_graph = input_networkx_graph_from_file(g_path)
        unweighted_graph = input_networkx_unweighted_graph_from_file(g_path)
        physical_graph = input_networkx_graph_from_file(pg_path)

        if output_dir.endswith('{}'):
            pass
        elif output_dir.endswith('/'):
            ...
            # output_dir += {}
        else:
            output_dir += '/{}'

        graph_name = g_path.split('/')[-1]
    case _:
        greed = Greed()
        greed.research()