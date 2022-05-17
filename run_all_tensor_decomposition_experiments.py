import subprocess
import datetime
import os

# NOTE: This script supports caching of previous results. If we kill the process
# then we should remove the incomplete log for the current run, so that it
# restarts from there.

def main():
    data = 'mri'
    ranks = ['1,1,1,1', '4,2,2,1', '8,2,2,1', '8,4,4,1', '8,4,4,2', '16,4,4,2']

    algorithm_list = ['HOOI', 'ALS', 'ALS-RS', 'ALS-DJSSW19']
    seeds = [0]

    max_num_steps = 5
    rre_gap_tol = 0

    samples_list = [1028, 4096]
    #alpha_list = [0.01]

    output_path = 'output/tensor_decomposition/{}/'.format(data)

    for rank in ranks:
        for samples in samples_list:
            for algorithm in algorithm_list:
                for seed in seeds:
                    pretty_rank = rank.replace(',', '.')
                    filename = 'alg{}-rank{}-seed{}-steps{}-tol{}'.format(algorithm, pretty_rank, seed, max_num_steps, rre_gap_tol)
                    if algorithm in ['ALS-RS', 'ALS-DJSSW19']:
                        filename += '-samples{}'.format(samples)
                    path = output_path + filename + '.txt'

                    print(path)
                    if os.path.exists(path):
                        print('Path already exists:', path)
                        continue

                    command = ['python3', 'tucker_decomposition_experiments.py']
                    command.append('--data={}'.format(data))
                    command.append('--algorithm={}'.format(algorithm))
                    command.append('--seed={}'.format(seed))
                    command.append('--rank={}'.format(rank))

                    command.append('--max_num_samples={}'.format(samples))
                    command.append('--max_num_steps={}'.format(max_num_steps))
                    command.append('--rre_gap_tol={}'.format(rre_gap_tol))

                    cmd = ' '.join(command)                                    
                    cmd += ' >> ' + path
                    print(cmd)
                    process = subprocess.Popen([cmd], shell=True)
                    process.wait()

main()

