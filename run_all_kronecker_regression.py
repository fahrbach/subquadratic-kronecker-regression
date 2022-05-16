import subprocess
import datetime

def main():
    ndim_list = [1, 2, 3]
    rows_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    cols_list = [2, 4, 8, 16, 32, 64, 128]
    alg_list = [1, 2, 3, 4, 5]
    seed_list = [0, 1, 2, 3, 4]

    alpha_list = [0.01]
    epsilon_list = [0.1]
    delta_list = [0.01]

    output_path = 'output/kronecker_regression/'

    for ndim in ndim_list:
        for rows in rows_list:
            for cols in cols_list:
                if cols > rows: continue
                for alg in alg_list:
                    for seed in seed_list:
                        for alpha in alpha_list:
                            for epsilon in epsilon_list:
                                for delta in delta_list:
                                    filename = 'alg{}-ndim{}-rows{}-cols{}-seed{}'.format(alg, ndim, rows, cols, seed)
                                    if alg in [4, 5]:
                                        filename += '-alpha{}'.format(alpha)
                                    path = output_path + filename + '.txt'

                                    command = ['python3', 'kronecker_regression_main.py']
                                    command.append('--ndim={}'.format(ndim))
                                    command.append('--rows={}'.format(rows))
                                    command.append('--cols={}'.format(cols))
                                    command.append('--alg={}'.format(alg))
                                    command.append('--seed={}'.format(seed))

                                    command.append('--alpha={}'.format(alpha))
                                    command.append('--epsilon={}'.format(epsilon))
                                    command.append('--delta={}'.format(delta))

                                    cmd = ' '.join(command)                                    
                                    cmd += ' >> ' + path
                                    print(cmd)
                                    process = subprocess.Popen([cmd], shell=True)
                                    process.wait()

main()

