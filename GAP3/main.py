from GAP import GAP, Node

from copy import deepcopy
from random import randint


class GAP3(GAP):
    def __init__(
            self,
            r1: float = 0.5,
            r2: float = 0.07,
            iter_max: int = 50,
            cut_ratio: float = 0.8,
            article: bool = False,
            graph_dirs: list[tuple[str, str]] | None = None,
            physical_graph_dirs: list[str] | None = None,
            iter_max_list: list[int] | None = None,
            r2_list: list[float] | None = None,
            r3_list: list[float] | None = None,
            cr_list: list[float] | None = None,
            num_of_tries: int = 3
        ) -> None:
        super().__init__(r1, r2, iter_max, cut_ratio, article)
        self.NAME = 'GAP3'
        self.NUM_OF_TRIES = num_of_tries

        if graph_dirs:
            self.graph_dirs = graph_dirs
        else:
            self.graph_dirs = [
                (r'./data/sausages', f'./results/{self.NAME}/sausages'),
                (r'./data/triangle/graphs', f'./results/{self.NAME}/triangle'),
            ]

        if physical_graph_dirs:
            self.physical_graph_dirs = physical_graph_dirs
        else:
            self.physical_graph_dirs = [
                r'./data/physical_graphs',
            ]
        
        if r3_list:
            self.r3_list = r3_list
        else:
            self.r3_list = [0.1]
            self.r3 = 0.1
        

        def mutate_partition(self, partition: dict[int, list[int]]) -> dict[int, list[int]]:
            new_partition = deepcopy(partition)
            for _ in range(self.r3 * len(self.G)):
                job = randint(0, len(self.G) - 1)
                proc = randint(0, self.k - 1)

                for key in new_partition:
                    if job in new_partition[key]:
                        new_partition[key].remove(job)
                        break

                new_partition[proc].append(job)
            
            return new_partition
        

        def gap(self, graph_path: str, physical_graph_path: str, if_do_load: bool = False, path: str = None, physical_graph_name: str = None) -> dict[int: list[int]]:
            print(graph_path, physical_graph_path)

            self.G = self.input_graph_from_file(graph_path)
            self.PG = self.input_graph_from_file(physical_graph_path)

            self.k = len(self.PG)

            ans = deepcopy(self.get_initial_partition(self.G, self.PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name))
            f_ans = self.f(ans)

            for _ in range(self.iter_max):
                partition_new = super().gap(self.G, self.PG, if_do_load=if_do_load, path=path, physical_graph_name=physical_graph_name, initial_partition=ans)
                if self.f(partition_new) < f_ans:
                    ans = deepcopy(partition_new)
                    f_ans = self.f(partition_new)

                partition_new = mutate_partition(self, ans)

                new_f = self.f(partition_new)
                if new_f < f_ans:
                    ans = partition_new
                    f_ans = new_f

            return ans

if __name__ == '__main__':
    print('GAP3')
    graph_dirs = [
        (r'./data/testing_graphs', './results/GAP3/testing_graphs'),
    ]
    gap_algo = GAP3(article=True, graph_dirs=graph_dirs, iter_max=100)
    gap_algo.research()
