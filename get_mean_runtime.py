import numpy as np
import matplotlib.pyplot as plt

def read_and_parse_log_file(filename):
    iterations = []
    with open(filename) as f:
        iteration_info = []
        for line in f.readlines():
            line = line.strip()
            tokens = line.split()
            if len(tokens) == 0: continue
            if line[0] == '#':   # Reset for new experiment.
                iterations = []
                iteration_info = []
            if tokens[0] == 'step:':
                if len(iteration_info) > 0:
                    iterations.append(iteration_info)
                iteration_info = []
            if tokens[0] == 'loss:':
                loss = eval(tokens[1])
                rmse = eval(tokens[3])
                time = eval(tokens[5])
                iteration_info.append([loss, rmse, time])
        if len(iteration_info) > 0:
            iteration_info.append([loss, rmse, time])
    return iterations

def main():
    shapes = [(512, 512, 512), (1024, 512, 512), (1024, 1024, 512), (1024, 1024, 1024)]
    ranks = [(2, 2, 2), (4, 2, 2), (4, 4, 2), (4, 4, 4)]
    algorithms = ['ALS', 'ALS-RS']
    for shape in shapes:
        for rank in ranks:
            for algorithm in algorithms:
                print(shape, rank, algorithm)

                filename = 'output/synthetic-all/synthetic-all'
                filename += '_' + ','.join([str(x) for x in shape])
                filename += '_' + ','.join([str(x) for x in rank])
                filename += '_' + algorithm
                filename += '.txt'

                iterations = read_and_parse_log_file(filename)
                #print(iterations)

                min_rmse = 10**10
                times = [[] for i in range(4)]
                for iteration in iterations:
                    idx = 0
                    for step in iteration:
                        loss, rmse, time = step
                        #print(rmse, time)
                        min_rmse = min(min_rmse, rmse)
                        times[idx].append(time)
                        idx += 1

                ans = []
                for i in range(4):
                    ans.append(np.mean(times[i]))
                ans.append(min_rmse)
                print(' '.join([str(x) for x in ans]))
                print()

main()
