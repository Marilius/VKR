#!/usr/bin/env python

import argparse

import random
import itertools

from dataclasses import dataclass


@dataclass
class Job:
    id: int
    proc: int
    length: float
    start_time: float
    end_time: float
    edges: list[int]


parser = argparse.ArgumentParser(description='Генератор графа по параметрам.')

parser.add_argument('-p', nargs="+", type=int, help='Производительности процессоров')
parser.add_argument('-L', type=int, help='Суммарная длительность работ на процессоре, одинаковая для каждого процессора.')
parser.add_argument('-min_l', type=int, help='Минимальная длительность работ на процессоре.')
parser.add_argument('-max_l', type=int, help='Максимальная длительность работ на процессоре.')
# parser.add_argument('-D', type=float, help='Плотность графа.')
parser.add_argument('-N_e', type=int, help='Число рёбер.')
parser.add_argument('-N_s', type=int, help='Число секущих рёбер.')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='Перемешивание номеров вершин.', default=False)

# парсим командную строку
args = parser.parse_args()
p = args.p
L = args.L
min_l = args.min_l
max_l = args.max_l
# D = args.D
N_e = args.N_e
N_s = args.N_s
shuffle = args.shuffle

# создание вершин графа
n0: int = 0
jobs: list[list[Job]] = [[] for _ in range(len(p))]
for i in range(len(p)):
    f = True
    while f:
        f = False
        n = n0
        time_left = L
        start_time = 0
        while time_left > 0:
            curr_time = random.randint(min_l, max_l)
            end_time = min(start_time + curr_time, L)
            curr_time = end_time - start_time
            
            if curr_time < min_l:
                f = True
                jobs[i] = []
                break

            jobs[i].append(Job(n, i, curr_time, start_time, end_time, []))
            n += 1

            start_time += curr_time
            time_left -= curr_time
            assert time_left >= 0

    n0 = n
exact_partition: list[int] = list(itertools.chain.from_iterable([[proc] * len(job_list) for proc, job_list in enumerate(jobs)]))
# print(exact_partition)
assert len(exact_partition) == len(list(itertools.chain.from_iterable(jobs)))

for i in range(len(p)):
    for job in jobs[i]:
        assert min_l <= job.length <= max_l
        assert 0 <= job.start_time < job.end_time <= L

for proc_jobs in jobs:
    weight = 0
    for job in proc_jobs:
        weight += job.length
    assert weight == L, f'{weight} != {L}'

# добавление не секущих рёбер
n = N_e - N_s
edge_list: list[list[int]] = [[] for _ in range(sum(map(len, jobs)))]
while n:
    proc_num = random.randint(0, len(jobs) - 1)
    
    while (first := random.choice(jobs[proc_num]).id) == (second := random.choice(jobs[proc_num]).id):
        ...
    
    first, second = min(first, second), max(first, second)

    if second not in edge_list[first]:
        assert first < second
        edge_list[first].append(second)
        n -= 1

n_all = 0
for i in jobs:
    n = 0
    for j in i:
        n += len(edge_list[j.id])
    n_all += n

assert n_all == N_e - N_s, f'{n_all} != {N_e - N_s}'


# добавление секущих рёбер
n = N_s
while n:
    while (proc_first := random.randint(0, len(jobs) - 1)) == (proc_second := random.randint(0, len(jobs) - 1)):
        ...
    
    first_job = random.choice(jobs[proc_first]) 
    second_job = random.choice(jobs[proc_second])
    
    if first_job.end_time <= second_job.start_time:
        first, second = first_job.id, second_job.id
    elif second_job.end_time <= first_job.start_time:
        first, second = second_job.id, first_job.id
    else:
        continue

    if second not in edge_list[first]:
        edge_list[first].append(second)
        n -= 1

n_all = 0
for i in jobs:
    n = 0
    for j in i:
        n += len(edge_list[j.id])
    n_all += n
assert n_all == N_e


if shuffle:
    jobs_ids = [job.id for job in itertools.chain.from_iterable(jobs)]
    # zipped = list(zip(jobs_ids, edge_list))
    # random.shuffle(zipped)
    # jobs_ids, edge_list = zip(*zipped)

    random.shuffle(jobs_ids)

    for job in itertools.chain.from_iterable(jobs):
        job.id = jobs_ids[job.id]
    # for job_edges in edge_list:
    #     for i in range(len(job_edges)):
    #         job_edges[i] = jobs_ids[job_edges[i]]
    assert len(set([job.id for job in itertools.chain.from_iterable(jobs)])) == len([job.id for job in itertools.chain.from_iterable(jobs)])

# запись в файл
NAME_FORMAT = "./data/gen_data/{p}_{L}_{min_l}_{max_l}_{N_e}_{N_s}.txt"
FORMAT = "{p}\n{L}\n{min_l} {max_l}\n{N_e} {N_s}\n{exact_partition}\n"
NODE_FORMAT = "{id} {weight} {child_list}\n"

name = NAME_FORMAT.format(
    p='_'.join(map(str, p)),
    L=L,
    min_l=min_l,
    max_l=max_l,
    N_e=N_e,
    N_s=N_s,
)

with open(name, 'w+') as f:
    f.write(FORMAT.format(
        p=' '.join(map(str, p)),
        L=L,
        min_l=min_l,
        max_l=max_l,
        N_e=N_e,
        N_s=N_s,
        exact_partition=' '.join(map(str, exact_partition))
    ))

    for proc_weight, proc_jobs in zip(p, jobs):
        for job in proc_jobs:
            f.write(NODE_FORMAT.format(
                id=job.id,
                weight=proc_weight * job.length,
                child_list=' '.join(map(str, sorted(edge_list[job.id])))
            ))
