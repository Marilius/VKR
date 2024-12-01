

paths = [
    # './results2/MK_greed_greed_with_geq_cr/weighted/testing_graphs/{}',
    './results2/MK_greed_greed_weighted/weighted/testing_graphs/{}',
    './results/greed/weighted/testing_graphs/{}',
    './results2/simple_part/weighted/testing_graphs/{}',
    
]

raise

test_graph_name = 'test.txt'

pgs = ['3_2x1.txt', '3_2x2.txt', '3_2x3.txt', '3_2x4.txt', '3_2x5.txt']
crs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]

# 3_2_3_2_3_2_3_2_3_2_1000_10_100_2.0_0.6_False.graph 3_2x5.txt 0.0 0.2 8333.333333333334 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
for i, path in enumerate(paths):
    # val = i + 1
    val = 1
    with open(path.format(test_graph_name), 'w') as f:
        print(path.format(test_graph_name))
        for j, pg in enumerate(pgs):
            # j += 1
            # val *= j
            for k, cr_max in enumerate(crs):
                if 'MK_greed_greed_weighted' in path:
                    val = cr_max
                # val *= cr_max
                f.write(
                    f'{test_graph_name} {pg} {cr_max} {cr_max} {val} {[]}\n'
                )
                
                
test_graph_name = 'test2.txt'
for i, path in enumerate(paths):
    with open(path.format(test_graph_name), 'w') as f:
        print(path.format(test_graph_name))
        for j, pg in enumerate(pgs):
            for k, cr_max in enumerate(crs):
                val = 1
                if 'MK_greed_greed_weighted' in path:
                    if cr_max in [0.15, 0.35, 0.5, 0.6]:
                        val = cr_max
                    if pg == '3_2x1.txt':
                        if cr_max in [0.2, 0.25, 0.4, 0.45, 0.65, 0.7]:
                            val = cr_max
                    if pg == '3_2x5.txt':
                        if cr_max in [0.4, 0.45]:
                            val = cr_max
                f.write(
                    f'{test_graph_name} {pg} {cr_max} {cr_max} {val} {[]}\n'
                )