import os.path
import numpy as np

# NOTE: Currently using seed=0 for all stats.

"""
Goal: Input fixed ndim and one of {rows, cols}.
Then output loss/time blocks that have been averaged over all seeds
"""
def main_fixed_cols():
    ndim = 2
    cols = 64

    rows_candidates = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    algorithms = [1, 3, 4, 5]
    seeds = [0]

    #rows_candidates = [1000, 2500, 5000, 7500, 10000]
    #algorithms = [1, 3, 4, 5]
    #seeds = [0]
    
    alpha = 1e-5
    #samples = 1028
    epsilon = 0.1
    delta = 0.01

    output_path = 'output/kronecker_regression/'

    time_by_rows_mean = []
    time_by_rows_std = []
    loss_by_rows_mean = []
    loss_by_rows_std = []
    for rows in rows_candidates:
        tmp_time_mean = []
        tmp_loss_mean = []
        for alg in algorithms:
            times = []
            losses = []
            files_exist = True
            for seed in seeds:
                filename = 'alg{}-ndim{}-rows{}-cols{}-seed{}'.format(alg, ndim, rows, cols, seed)
                if alg in [4, 5]:
                    filename += '-alpha{}'.format(alpha)
                path = output_path + filename + '.txt'

                if not os.path.exists(path):
                    files_exist = False
                    continue
                with open(path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line[:4] == 'time':
                            #print(line)
                            times.append(eval(line.split()[1]))
                        if line[:4] == 'loss':
                            #print(line)
                            losses.append(eval(line.split()[1]))
            if len(times) != len(seeds):
                print('Missing times:', rows, alg, times)
            if len(losses) != len(seeds):
                print('Missing losses:', rows, alg, times)

            print(rows, alg, times)
            print(rows, alg, losses)

            if len(times) == 0:
                if files_exist:
                    tmp_time_mean.append('-')
                else:
                    tmp_time_mean.append('.')
            else:
                tmp_time_mean.append(np.mean(times))

            if len(losses) == 0:
                if files_exist:
                    tmp_loss_mean.append('-')
                else:
                    tmp_loss_mean.append('.')
            else:
                tmp_loss_mean.append(np.mean(losses))

        time_by_rows_mean.append(tmp_time_mean)
        loss_by_rows_mean.append(tmp_loss_mean)

    print('\nndim={} cols={}'.format(ndim, cols))

    # Print formatted output for running time.
    print('\nRunning times:')
    for row in time_by_rows_mean:
        formatted = ' '.join(str(x) for x in row)
        print(formatted)

    # Print formatted output for losses.
    print('\nLosses:')
    for row in loss_by_rows_mean:
        formatted = ' '.join(str(x) for x in row)
        print(formatted)

"""
Goal: Input fixed ndim and one of {rows, cols}.
Then output loss/time blocks that have been averaged over all seeds
"""
def main_fixed_rows():
    ndim = 2
    rows = 16384

    cols_candidates = [2, 4, 8, 16, 32, 64, 128]
    algorithms = [1, 2, 3, 4, 5]
    seeds = [0]
    
    #alpha = 0.01
    samples = 1028
    epsilon = 0.1
    delta = 0.01

    output_path = 'output/kronecker_regression/'

    time_by_rows_mean = []
    time_by_rows_std = []
    loss_by_rows_mean = []
    loss_by_rows_std = []
    for cols in cols_candidates:
        tmp_time_mean = []
        tmp_loss_mean = []
        for alg in algorithms:
            times = []
            losses = []
            files_exist = True
            for seed in seeds:
                filename = 'alg{}-ndim{}-rows{}-cols{}-seed{}'.format(alg, ndim, rows, cols, seed)
                if alg in [4, 5]:
                    filename += '-samples{}'.format(samples)
                path = output_path + filename + '.txt'

                if not os.path.exists(path):
                    files_exist = False
                    continue
                with open(path, 'r') as f:
                    last_time_seen = -1  # NOTE: Read the last logged time.
                    last_loss_seen = -1
                    for line in f.readlines():
                        line = line.strip()
                        if line[:4] == 'time':
                            last_time_seen = eval(line.split()[1])
                        if line[:4] == 'loss':
                            last_loss_seen = eval(line.split()[1])
                    if last_time_seen != -1:
                        times.append(last_time_seen)
                    if last_loss_seen != -1:
                        losses.append(last_loss_seen)
            if len(times) != len(seeds):
                print('Missing times:', rows, alg, times)
            if len(losses) != len(seeds):
                print('Missing losses:', rows, alg, times)

            print(rows, alg, times)
            print(rows, alg, losses)

            if len(times) == 0:
                if files_exist:
                    tmp_time_mean.append('-')
                else:
                    tmp_time_mean.append('.')
            else:
                tmp_time_mean.append(np.mean(times))

            if len(losses) == 0:
                if files_exist:
                    tmp_loss_mean.append('-')
                else:
                    tmp_loss_mean.append('.')
            else:
                tmp_loss_mean.append(np.mean(losses))

        time_by_rows_mean.append(tmp_time_mean)
        loss_by_rows_mean.append(tmp_loss_mean)

    print('\nndim={} rows={}'.format(ndim, rows))

    # Print formatted output for running time.
    print('\nRunning times:')
    for row in time_by_rows_mean:
        formatted = ' '.join(str(x) for x in row)
        print(formatted)

    # Print formatted output for losses.
    print('\nLosses:')
    for row in loss_by_rows_mean:
        formatted = ' '.join(str(x) for x in row)
        print(formatted)

def main():
    main_fixed_cols()
    #main_fixed_rows()

main()
