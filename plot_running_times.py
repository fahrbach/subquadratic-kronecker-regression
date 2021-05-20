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
    filename_to_label_map = {}
    filename_to_label_map['output/Cardiac_MRI_data/sol_yxzt_pat1_ALS_4,4,2,2_10.txt'] = 'ALS'
    filename_to_label_map['output/Cardiac_MRI_data/sol_yxzt_pat1_ALS-RS_4,4,2,2_20.txt'] = 'ALS-RS'
    min_iteration = 0

    iterations = {}
    for filename in filename_to_label_map:
        iterations[filename] = read_and_parse_log_file(filename)

    for filename in filename_to_label_map:
        label = filename_to_label_map[filename]
        x_time = []
        y_rmse = []
        total_time = 0.0
        iteration_counter = 0
        for iteration in iterations[filename]:
            for step in iteration:
                loss, rmse, time = step
                total_time += time
                if iteration_counter >= min_iteration:
                    x_time.append(total_time)
                    y_rmse.append(rmse)
            iteration_counter += 1
        plt.plot(x_time, y_rmse, label=label)
    plt.legend()
    plt.show()

main()
