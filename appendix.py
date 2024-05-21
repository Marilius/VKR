from helpers import input_networkx_graph_from_file


graphs = [
    'testing_graphs/16_envelope_mk_eq.time',
    'testing_graphs/16_envelope_mk_rand.time',
    # 'testing_graphs/64_envelope_mk_eq.time',
    # 'testing_graphs/64_envelope_mk_rand.time',
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

lines = []
with open('appendix.txt', 'w') as f:
    for graph in graphs:
        path = f'./data/{graph}'.replace('.time', '.txt').replace('triangle', 'triangle/graphs')
        g = input_networkx_graph_from_file(path)
        # print(f'\item ${g.graph['graph_name']}$: {len(g.nodes())} вершин, {len(g.edges())} ребер;')

        lines += ['\\begin{figure}[H]']
        lines += ['\t\\centering']
        lines += [f'\t\\includegraphics[width=\\textwidth]{{./appendix_pics/{g.graph["graph_name"]}_3_2.png}}']
        lines += [f"\t\\caption{{Результаты для графа {g.graph['graph_name'].replace("_", "\\_")} ({len(g.nodes())} вершин, {len(g.edges())} рёбер)}}"]
        lines += [f'\t\\label{{fig:TK_{g.graph['graph_name']}_3_2}}']
        lines += ['\\end{figure}']
        lines += ['']

        lines += ['\\begin{figure}[H]']
        lines += ['\t\\centering']
        lines += [f'\t\\includegraphics[width=\\textwidth]{{./appendix_pics/{g.graph["graph_name"]}_4_1.png}}']
        lines += [f"\t\\caption{{Результаты для графа {g.graph['graph_name'].replace("_", "\\_")} ({len(g.nodes())} вершин, {len(g.edges())} рёбер)}}"]
        lines += [f'\t\\label{{fig:TK_{g.graph['graph_name']}_4_1}}']
        lines += ['\\end{figure}']
        lines += ['']

        lines += ['\\begin{figure}[H]']
        lines += ['\t\\centering']
        lines += [f'\t\\includegraphics[width=\\textwidth]{{./appendix_pics/{g.graph["graph_name"]}_5_4_3_2.png}}']
        lines += [f"\t\\caption{{Результаты для графа {g.graph['graph_name'].replace("_", "\\_")} ({len(g.nodes())} вершин, {len(g.edges())} рёбер)}}"]
        lines += [f'\t\\label{{fig:TK_{g.graph['graph_name']}_5_4_3_2}}']
        lines += ['\\end{figure}']
        lines += ['']

        #     <теплокарта>
        # Рис. ХХХ. Результаты для на прямоугольного графа (FIXME вершин, FIXME рёбер, веса вершин одинаковые).
        # // это образец; тут совсем автогеном не обойтись

    f.write('\n'.join(lines))