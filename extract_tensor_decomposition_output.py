import os.path
import numpy as np

def read_hooi_rre(path):
    with open(path) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) == 0: continue
            if tokens[0] == 'rre:':
                return tokens[1]
    return '-'

def read_rre(path):
    rre_list = []
    with open(path) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) == 0: continue
            if tokens[0] != 'loss:': continue
            rre = tokens[5]
            rre_list.append(rre)
    return rre_list[-1]

def read_hooi_time(path):
    with open(path) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) == 0: continue
            if tokens[0] == 'avg':
                return tokens[3]
    return '-'

def read_time(path):
    time_list = []
    num_steps = -1
    with open(path) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) == 0: continue
            if tokens[0] == 'step:':
                num_steps = int(tokens[1])
            if tokens[0] == 'loss:':
                t = eval(tokens[-1])
                time_list.append(t)
    return str(np.sum(time_list) / num_steps)

def Run():
    data = 'mri'
    ranks = ['1,1,1,1', '4,2,2,1', '4,4,2,2', '8,2,2,1', '8,4,4,1', '8,4,4,2', '8,8,2,2', '8,8,4,4', '16,4,4,2', '16,16,4,4']

    #data = 'coil'
    #ranks = ['1,1,1,1', '4,2,2,1', '8,2,2,1', '8,4,4,1', '8,4,4,2', '16,4,4,2']

    #data = 'hyperspectral'
    #ranks = ['1,1,1', '2,2,2', '3,3,3', '4,4,4', '5,5,5', '8,8,4', '8,8,8', '16,16,4']

    algorithms = ['HOOI', 'ALS', 'ALS-RS', 'ALS-DJSSW19']
    seeds = [0]

    max_num_steps = 5
    rre_gap_tol = 0
    samples_list = [1028, 4096, 16384]

    output_path = 'output/tensor_decomposition/{}/'.format(data)

    final_formated_rre = []
    final_formated_time = []

    for rank in ranks:
        final_formated_rre.append([rank])
        final_formated_time.append([rank])
        for algorithm in algorithms:
            for samples in samples_list:
                if algorithm in ['HOOI', 'ALS'] and samples != samples_list[0]: continue
                for seed in seeds:
                    print(rank)
                    pretty_rank = rank.replace(',', '.')
                    filename = 'alg{}-rank{}-seed{}-steps{}-tol{}'.format(algorithm, pretty_rank, seed, max_num_steps, rre_gap_tol)
                    if algorithm in ['ALS-RS', 'ALS-DJSSW19']:
                        filename += '-samples{}'.format(samples)
                    path = output_path + filename + '.txt'
                    print(path)
                    if not os.path.exists(path):
                        final_formated_rre[-1].append('.')
                        final_formated_time[-1].append('.')
                        continue
                    if algorithm == 'HOOI':
                        rre = read_hooi_rre(path)
                        time = read_hooi_time(path)
                    else:
                        rre = read_rre(path)
                        time = read_time(path)
                    final_formated_rre[-1].append(rre)
                    final_formated_time[-1].append(time)

    # Output
    print()
    print('data: {} rre'.format(data))
    for row in final_formated_rre:
        print(' '.join(row))

    print('\ndata: {} time'.format(data))
    for row in final_formated_time:
        print(' '.join(row))

def main():
    Run()

main()
