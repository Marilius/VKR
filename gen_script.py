import subprocess

# Arguments to be passed to the called script
# args_names = ['-p', '-L', '-min_l', '-max_l', '-N', '-cr', '-n_tries']

p_list = [
    [3, 2], [3, 2] * 2, [3, 2] * 3, [3, 2] * 4, [3, 2] * 5,
    [4, 1], [4, 1] * 2, [4, 1] * 3, [4, 1] * 4, [4, 1] * 5,
    [5, 4, 3, 2], [5, 4, 3, 2] * 2, [5, 4, 3, 2] * 3, [5, 4, 3, 2] * 4, [5, 4, 3, 2] * 5, 
]

L_list = [100, 500, 1000, 2000, 3000, 4000]

min_l = 10
max_l = 100

N_list = [1.5, 1.6, 1.7, 1.8, 1.9, 2, 3]

cr_list = [0.1, 0.2, 0.3, 0.4]

for p in p_list:
    for L in L_list:
        for N in N_list:
            for cr in cr_list:
                command = f'poetry run python ./task_graph_generator.py -p {" ".join(map(str, p))} -L {L} -min_l {min_l} -max_l {max_l} -N {N} -cr {cr} -n_tries 10000'
                # print(command)
                subprocess.run(command.split())
                # subprocess.run(command)


# Run the called script with arguments
