import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.datasets import synthetic
from tensorly.random import random_tucker
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import skvideo.io

output_file = None


def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)


def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)


def solve_least_squares(A, b, l2_regularization):
    AtA_lambda = A.T @ A + l2_regularization * np.identity(A.shape[1])
    Atb = A.T @ b
    return np.linalg.solve(AtA_lambda, Atb)


# TODO(fahrbach): Show why higher-order SVD methods do not scale well? Seems
# that they might if we use compressed representation?
def update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization, debug_mode):
    # See the matricized version of Equation 4.2 in "Tensor Decompositions and
    # Applications."
    Y_matrix = tl.unfold(Y_tensor, factor_index)
    core_matrix = tl.unfold(X_tucker.core, factor_index)
    K_matrix = np.identity(1)
    for n in range(X_tucker.core.ndim):
        if n == factor_index: continue
        # Important: For some reason, the ordering of this Kronecker product
        # disagrees with Equation 4.2 in the reference above. Did they introduce
        # a typo by thinking that the transpose reverses the order?
        K_matrix = np.kron(K_matrix, X_tucker.factors[n])
    # Each row of the current factor matrix is its own least squares problem.
    design_matrix = K_matrix @ core_matrix.T
    if debug_mode:
        print(' - design matrix shape:', design_matrix.shape)
        print(' - number of least squares solves:', X_tucker.factors[factor_index].shape[0])

    # Use the same "normalized" design matrix for all least squares solves.
    AtA_lambda = design_matrix.T @ design_matrix \
                 + l2_regularization * np.identity(design_matrix.shape[1])
    for row_index in range(X_tucker.factors[factor_index].shape[0]):
        response_vec = Y_matrix[row_index, :]
        Atb = design_matrix.T @ response_vec
        X_tucker.factors[factor_index][row_index, :] = \
            np.linalg.solve(AtA_lambda, Atb)


# Naive core tensor update that explicitly constructs the design matrix. This
# requires O((I_1 * I_2 * I_3) * (R_1 * R_2 * R_3)) space, and is prohibitively
# expensive for anything interesting. It also seems less numerically stable?
def update_core_tensor_naive(X_tucker, Y_tensor, l2_regularization):
    design_matrix = np.identity(1)
    for n in range(X_tucker.core.ndim):
        design_matrix = np.kron(design_matrix, X_tucker.factors[n])

    Y_vec = tl.tensor_to_vec(Y_tensor)

    new_core_tensor_vec = solve_least_squares( \
        design_matrix, Y_vec, l2_regularization)
    X_tucker.core = tl.reshape(new_core_tensor_vec, X_tucker.core.shape)


# Memory-efficient construction of the normal equation for the core tensor
# update.
#
# Note that we manually sped up the computation of K^T * vec(Y) by storing
# partially constructed Kronecker rows to avoid recomputation. This gave a
# speedup of ~40% over the simpler implementation.
def update_core_tensor_memory_efficient(X_tucker, Y_tensor, l2_regularization, debug_mode):
    start_time = time.time()
    KtK_lambda = np.identity(1)
    for n in range(len(X_tucker.factors)):
        KtK_lambda = np.kron(KtK_lambda, \
                             X_tucker.factors[n].T @ X_tucker.factors[n])
    KtK_lambda += l2_regularization * np.identity(KtK_lambda.shape[0])
    end_time = time.time()
    if debug_mode:
        print(' - KtK_lambda construction time:', end_time - start_time)

    factors_T = [factor.T.copy() for factor in X_tucker.factors]
    Y_vec = tl.tensor_to_vec(Y_tensor)
    b = np.zeros(KtK_lambda.shape[0])
    row_index = 0
    if debug_mode:
        num_rows = len(Y_vec)
        num_cols = KtK_lambda.shape[0]
        print(' - design matrix shape:', (num_rows, num_cols))

    # Note: fahrbach thinks this is still much slower than it could be due to
    # Python loop slowdowns. Is it possible to only use NumPY operations without
    # using too much memory?
    start_time = time.time()
    if X_tucker.core.ndim == 2:
        for core_index_0 in range(X_tucker.core.shape[0]):
            factor_matrix_col_T_0 = X_tucker.factors[0][:, core_index_0].T
            design_matrix_col_T_0 = factor_matrix_col_T_0
            for core_index_1 in range(X_tucker.core.shape[1]):
                factor_matrix_col_T_1 = X_tucker.factors[1][:, core_index_1].T
                design_matrix_col_T_1 = np.kron(design_matrix_col_T_0, factor_matrix_col_T_1)
                b[row_index] = design_matrix_col_T_1 @ Y_vec
                row_index += 1

    elif X_tucker.core.ndim == 3:
        for core_index_0 in range(X_tucker.core.shape[0]):
            factor_matrix_col_T_0 = X_tucker.factors[0][:, core_index_0].T
            design_matrix_col_T_0 = factor_matrix_col_T_0
            for core_index_1 in range(X_tucker.core.shape[1]):
                factor_matrix_col_T_1 = X_tucker.factors[1][:, core_index_1].T
                design_matrix_col_T_1 = np.kron(design_matrix_col_T_0, factor_matrix_col_T_1)
                for core_index_2 in range(X_tucker.core.shape[2]):
                    factor_matrix_col_T_2 = X_tucker.factors[2][:, core_index_2].T
                    design_matrix_col_T_2 = np.kron(design_matrix_col_T_1, factor_matrix_col_T_2)
                    b[row_index] = design_matrix_col_T_2 @ Y_vec
                    row_index += 1

    elif X_tucker.core.ndim == 4:
        for core_index_0 in range(X_tucker.core.shape[0]):
            factor_matrix_col_T_0 = X_tucker.factors[0][:, core_index_0].T
            design_matrix_col_T_0 = factor_matrix_col_T_0
            for core_index_1 in range(X_tucker.core.shape[1]):
                factor_matrix_col_T_1 = X_tucker.factors[1][:, core_index_1].T
                design_matrix_col_T_1 = np.kron(design_matrix_col_T_0, factor_matrix_col_T_1)
                for core_index_2 in range(X_tucker.core.shape[2]):
                    factor_matrix_col_T_2 = X_tucker.factors[2][:, core_index_2].T
                    design_matrix_col_T_2 = np.kron(design_matrix_col_T_1, factor_matrix_col_T_2)
                    for core_index_3 in range(X_tucker.core.shape[3]):
                        factor_matrix_col_T_3 = X_tucker.factors[3][:, core_index_3].T
                        design_matrix_col_T_3 = np.kron(design_matrix_col_T_2, factor_matrix_col_T_3)
                        b[row_index] = design_matrix_col_T_3 @ Y_vec
                        row_index += 1
    else:
        print('Core tensor of order', X_tucker.core.ndim, 'not supported.')
        assert (False)
    end_time = time.time()
    if debug_mode:
        print(' - Ktb construction time:', end_time - start_time)

    start_time = time.time()
    new_core_tensor_vec = np.linalg.solve(KtK_lambda, b)
    end_time = time.time()
    if debug_mode:
        print(' - np.linalg.solve(KtK_lambda, b) time:', end_time - start_time)
    X_tucker.core = tl.reshape(new_core_tensor_vec, X_tucker.core.shape)


def compute_ridge_leverage_scores(A, l2_regularization):
    normal_matrix = A.T @ A + l2_regularization * np.identity(A.shape[1])
    normal_matrix_pinv = np.linalg.pinv(normal_matrix)
    leverage_scores = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        leverage_scores[i] = A[i, :] @ normal_matrix_pinv @ A[i, :].T
    return leverage_scores


def write_leverage_scores_to_file(leverage_scores, X_tucker, l2_regularization, epsilon, delta, step, alpha):
    filename = 'leverage_scores.txt'
    with open(filename, 'w') as f:
        instance_info = [X_tucker.core.ndim, l2_regularization, epsilon, delta, step, alpha]
        f.write(' '.join(str(_) for _ in instance_info) + '\n')
        for n in range(X_tucker.core.ndim):
            factor = X_tucker.factors[n]
            factor_spectral_norm = np.linalg.norm(factor, 2) ** 2
            factor_info = [factor.shape[0], factor.shape[1], factor_spectral_norm]
            f.write(' '.join(str(_) for _ in factor_info) + '\n')
            f.write(' '.join(str(score) for score in leverage_scores[n]) + '\n')


def update_core_tensor_by_row_sampling(X_tucker, Y_tensor, l2_regularization, step, epsilon, delta, downsampling_ratio,
                                       debug_mode):
    global output_file

    start_time = time.time()
    leverage_scores = [compute_ridge_leverage_scores(factor, 0.0) for factor in X_tucker.factors]
    end_time = time.time()
    if debug_mode:
        print(' - leverage score computation time:', end_time - start_time)

    num_original_rows = 1
    num_augmented_rows = 1
    for n in range(X_tucker.core.ndim):
        num_original_rows *= X_tucker.factors[n].shape[0]
        num_augmented_rows *= X_tucker.factors[n].shape[1]

    num_core_elements = 1
    for dimension in X_tucker.core.shape:
        num_core_elements *= dimension

    start_time = time.time()
    write_leverage_scores_to_file(leverage_scores, X_tucker, l2_regularization, epsilon, delta, step,
                                  downsampling_ratio)
    cmd = './row_sampling'
    os.system(cmd)
    end_time = time.time()
    if debug_mode:
        print(' - row sampling subroutine time:', end_time - start_time)

    filename = 'sampled_rows.csv'
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_merged_rows, num_sampled_rows = [int(_) for _ in lines[0].split(',')]
        assert (len(lines) == num_merged_rows + 1)
        if debug_mode:
            print(' - num sampled rows:', num_sampled_rows)
            print(' - num merged rows:', num_merged_rows, \
                  'ratio:', float(num_merged_rows) / (num_original_rows + num_augmented_rows))
            output_file.write('num_sampled_rows: {} num_merged_rows: {} ratio: {}\n'.format( \
                num_sampled_rows, num_merged_rows, float(num_merged_rows) / (num_original_rows + num_augmented_rows)))

        # Memory-efficient core tensor step
        start_time = time.time()
        # Note: This is not memory efficient, but I want to get things off the ground.
        design_matrix = np.zeros((num_merged_rows, num_core_elements))
        SAtSA = np.zeros((num_core_elements, num_core_elements))
        SAtb = np.zeros((num_core_elements, 1))

        if debug_mode:
            print(' - sampled augmented design matrix shape:', design_matrix.shape)
        for line_index in range(1, len(lines)):
            line = lines[line_index].strip().split(',')
            sketched_row_index = line_index - 1
            shape_indices = [int(_) for _ in line[:-2]]
            sample_probability = float(line[-2])
            sample_weight = float(line[-1])  # We merge repeated samples.
            # print(' *', shape_indices, sample_probability, sample_weight)

            # Construct rows in the augmented, sampled design matrix.
            if shape_indices[0] == -1:  # Encoding for ridge rows.
                row = np.zeros((1, num_core_elements))
                row[0, shape_indices[1]] = l2_regularization ** 0.5
            else:
                # Note: It seems that constructing this row is fairly slow...
                row = np.identity(1)
                for n in range(X_tucker.core.ndim):
                    row = np.kron(row, X_tucker.factors[n][shape_indices[n], :])
            rescaling_coeff = (sample_weight / num_sampled_rows) / sample_probability
            SAtSA += rescaling_coeff * row.T @ row

            # Construct entries in the augmented, sampled response vector.
            sketched_responce_value = 0
            if shape_indices[0] == -1:
                sketched_responce_value = 0
            else:
                sketched_responce_value = Y_tensor[tuple(shape_indices)]
                sketched_responce_value *= rescaling_coeff
            SAtb += sketched_responce_value * row.T
        end_time = time.time()
        if debug_mode:
            print(' - sampled least squares construction time:', end_time - start_time)

        new_core_vec = np.linalg.solve(SAtSA, SAtb)
        X_tucker.core = tl.reshape(new_core_vec, X_tucker.core.shape)


def compute_loss(Y_tensor, X_tucker, l2_regularization):
    loss = 0.0
    residual_vec = tl.tensor_to_vec(Y_tensor - tl.tucker_to_tensor(X_tucker))
    loss += np.linalg.norm(residual_vec) ** 2
    loss += l2_regularization * np.linalg.norm(tl.tensor_to_vec(X_tucker.core)) ** 2
    for n in range(len(X_tucker.factors)):
        loss += l2_regularization * np.linalg.norm(X_tucker.factors[n]) ** 2
    return loss


def run_alternating_least_squares(X_tucker, Y_tensor, l2_regularization, \
                                  algorithm, num_steps, epsilon, delta, downsampling_ratio, debug_mode):
    global output_file

    num_elements = 1
    for n in X_tucker.shape:
        num_elements *= n
    loss = compute_loss(Y_tensor, X_tucker, l2_regularization)

    for step in range(num_steps):
        print('step:', step)
        output_file.write('step: ' + str(step) + '\n')
        output_file.flush()

        # --------------------------------------------------------------------------
        # Factor matrix updates.
        for factor_index in range(X_tucker.core.ndim):
            if debug_mode:
                print('Updating factor matrix:', factor_index)
            start_time = time.time()
            update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization, debug_mode)
            end_time = time.time()

            new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
            rmse = (new_loss / num_elements) ** 0.5
            print('loss: {} RMSE: {} time: {}'.format(new_loss, rmse, end_time - start_time))
            output_file.write('loss: {} RMSE: {} time: {}'.format(new_loss, rmse, end_time - start_time) + '\n')
            if debug_mode and new_loss > loss:
                print('Warning: The loss function increased!')
                output_file.write('Warning: The loss function increased!\n')
            output_file.flush()
            loss = new_loss

        # --------------------------------------------------------------------------
        # Core tensor update.
        if debug_mode:
            print('Updating core tensor:')
        start_time = time.time()
        if algorithm == 'ALS':
            # update_core_tensor_naive(X_tucker, Y_tensor, l2_regularization)
            update_core_tensor_memory_efficient(X_tucker, Y_tensor, l2_regularization, debug_mode)
        elif algorithm == 'ALS-RS':
            update_core_tensor_by_row_sampling(X_tucker, Y_tensor, l2_regularization, step, epsilon, delta,
                                               downsampling_ratio, debug_mode)
        else:
            print('algorithm:', algorithm, 'is unsupported.')
            assert (False)
        end_time = time.time()

        new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
        rmse = (new_loss / num_elements) ** 0.5
        print('loss: {} RMSE: {} time: {}'.format(new_loss, rmse, end_time - start_time))
        output_file.write('loss: {} RMSE: {} time: {}'.format(new_loss, rmse, end_time - start_time) + '\n')
        if debug_mode and new_loss > loss:
            print('Warning: The loss function increased!')
            output_file.write('Warning: The loss function increased!\n')
        output_file.flush()
        loss = new_loss
        print()


# ==============================================================================
# Synthetic Experiment 1:
# - Simple tensor decomposition experiment where a tensor Y is randomly generated
#   by a random Tucker decomposition, with one entry set to Y[0,0,0] = 1, so
#   that it can't be fit perfectly.
# - Then we generate a new random Tucker decomposition X (using a different
#   seed), and we try to learn Y.
# - Note: We start to see nice gains from ALG-RS when the tensor has shape
#   ~(1028, 1028, 512) and the rank is (4, 4, 4).
# ==============================================================================
def run_synthetic_experiment_1():
    shape = (1028, 512, 512)
    rank = (8, 4, 2)
    steps = 10
    l2_regularization = 0.001
    seed = 0
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS'
    # algorithm = 'ALS-RS'

    global output_file
    output_filename = 'output/synthetic-1/synthetic-1'
    output_filename += '_' + ','.join([str(x) for x in shape])
    output_filename += '_' + ','.join([str(x) for x in rank])
    output_filename += '_' + algorithm
    output_filename += '.txt'

    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')

    # Initialize target tensor Y.
    Y_tucker = random_tucker(shape, rank, random_state=(seed + 1000))
    Y = tl.tucker_to_tensor(Y_tucker)
    Y[0, 0, 0] = 1

    print('Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')

    print('shape: ', shape)
    output_file.write('rank: ' + str(rank) + '\n')
    print('rank: ', rank)
    output_file.write('rank: ' + str(rank) + '\n')
    print('seed: ', seed)
    output_file.write('seed: ' + str(seed) + '\n')
    print('l2_regularization: ', l2_regularization)
    output_file.write('l2_regularization: ' + str(l2_regularization) + '\n')
    print('steps: ', steps)
    output_file.write('steps: ' + str(steps) + '\n')
    print('epsilon: ', epsilon)
    output_file.write('epsilon: ' + str(epsilon) + '\n')
    print('delta: ', delta)
    output_file.write('delta: ' + str(delta) + '\n')
    print('downsampling_ratio: ', downsampling_ratio)
    output_file.write('downsampling_ratio: ' + str(downsampling_ratio) + '\n')
    print('algorithm: ', algorithm)
    output_file.write('algorithm: ' + str(algorithm) + '\n')
    output_file.flush()

    X_tucker = random_tucker(Y.shape, rank, random_state=seed)
    if algorithm in ['ALS', 'ALS-RS']:
        os.system('g++-10 -O2 -std=c++11 row_sampling.cc -o row_sampling')
        run_alternating_least_squares(X_tucker, Y, l2_regularization, algorithm, steps, epsilon, delta,
                                      downsampling_ratio, True)

    X = tl.tucker_to_tensor(X_tucker)
    print(X)


# ==============================================================================
# Synthetic Shapes Experiment:
# - Use Tensorly's built-in shape images.
# - Note: This data can easily scale up, and starts to show the benefit of
#   row sampling. For example, create a shape of dimensions [1024, 1024, 3]
#   and rank [4, 4, 3]. Observe that it's only sampling about 0.1% of the rows.
# ==============================================================================
def run_synthetic_shapes_experiment():
    pattern = 'circle'  # ['rectangle', 'swiss', 'circle']
    n = 2048
    rank = [4, 4, 2]
    steps = 10
    l2_regularization = 0.001
    seed = 0
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS-RS'
    # algorithm = 'ALS-RS'

    global output_file
    output_filename = 'output/synthetic-shapes/synthetic_shapes'
    output_filename += '_' + pattern
    output_filename += '_' + str(n)
    output_filename += '_' + ','.join([str(x) for x in rank])
    output_filename += '_' + algorithm
    output_filename += '.txt'

    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')

    # Initialize target tensor Y.
    Y = tl.datasets.synthetic.gen_image(pattern, n, n, 3)
    plt.imshow(Y)
    plt.show()

    print('Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')

    print('n: ', n)
    output_file.write('rank: ' + str(rank) + '\n')
    print('rank: ', rank)
    output_file.write('rank: ' + str(rank) + '\n')
    print('seed: ', seed)
    output_file.write('seed: ' + str(seed) + '\n')
    print('l2_regularization: ', l2_regularization)
    output_file.write('l2_regularization: ' + str(l2_regularization) + '\n')
    print('steps: ', steps)
    output_file.write('steps: ' + str(steps) + '\n')
    print('epsilon: ', epsilon)
    output_file.write('epsilon: ' + str(epsilon) + '\n')
    print('delta: ', delta)
    output_file.write('delta: ' + str(delta) + '\n')
    print('downsampling_ratio: ', downsampling_ratio)
    output_file.write('downsampling_ratio: ' + str(downsampling_ratio) + '\n')
    print('algorithm: ', algorithm)
    output_file.write('algorithm: ' + str(algorithm) + '\n')
    output_file.flush()

    X_tucker = random_tucker(Y.shape, rank, random_state=seed)
    if algorithm in ['ALS', 'ALS-RS']:
        os.system('g++-10 -O2 -std=c++11 row_sampling.cc -o row_sampling')
        run_alternating_least_squares(X_tucker, Y, l2_regularization, algorithm, steps, epsilon, delta,
                                      downsampling_ratio, True)

    X = tl.tucker_to_tensor(X_tucker)
    plt.imshow(X)
    plt.show()


def create_output_filename(input_filename, algorithm, rank, steps):
    # Remove "data/" prefix.
    name = input_filename[5:].split('.')[0]
    output_filename = 'output/' + name
    output_filename += '_' + algorithm
    output_filename += '_' + ','.join([str(x) for x in rank])
    output_filename += '_' + str(steps)
    output_filename += '.txt'
    return output_filename


# ==============================================================================
# Cardiac MRI Experiment:
# - Read 4-way tensor with shape (256, 256, 14, 20), which corresponds to
#   positions (x, y, z, time), and run ALS with and without row sampling.
# ==============================================================================
def run_cardiac_mri_experiment():
    input_filename = 'data/Cardiac_MRI_data/sol_yxzt_pat1.mat'
    Y = sio.loadmat(input_filename)['sol_yxzt']

    algorithm = 'ALS-RS'
    # algorithm = 'ALS'
    rank = (4, 4, 2, 2)
    seed = 0
    l2_regularization = 0.001
    steps = 3
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0

    global output_file
    output_filename = create_output_filename(input_filename, algorithm, rank, steps)
    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')
    print('input_filename: ', input_filename)
    output_file.write('input_filename: ' + input_filename + '\n')

    print('Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')

    print('rank: ', rank)
    output_file.write('rank: ' + str(rank) + '\n')
    print('seed: ', seed)
    output_file.write('seed: ' + str(seed) + '\n')
    print('algorithm: ', algorithm)
    output_file.write('algorithm: ' + str(algorithm) + '\n')
    print('l2_regularization: ', l2_regularization)
    output_file.write('l2_regularization: ' + str(l2_regularization) + '\n')
    print('steps: ', steps)
    output_file.write('steps: ' + str(steps) + '\n')
    print('epsilon: ', epsilon)
    output_file.write('epsilon: ' + str(epsilon) + '\n')
    print('delta: ', delta)
    output_file.write('delta: ' + str(delta) + '\n')
    print('downsampling_ratio: ', downsampling_ratio)
    output_file.write('downsampling_ratio: ' + str(downsampling_ratio) + '\n')
    output_file.flush()

    X_tucker = random_tucker(Y.shape, rank, random_state=seed)
    if algorithm in ['ALS', 'ALS-RS']:
        os.system('g++-10 -O2 -std=c++11 row_sampling.cc -o row_sampling')
        run_alternating_least_squares(X_tucker, Y, l2_regularization, algorithm, steps, epsilon, delta,
                                      downsampling_ratio, True)

    X = tl.tucker_to_tensor(X_tucker)
    print(X)


# ==============================================================================
# Image Experiments
# - Reads an image as a 3-way tensor (x, y, RGB channel), and 
# ==============================================================================
def run_image_experiment():
    input_filename = 'data/images/nyc.jpg'

    dimensions = [1024, 1024, 3]
    rank = [4, 4, 2]
    seed = 0
    l2_regularization = 0.001
    steps = 10
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS-RS'
    # algorithm = 'ALS'

    global output_file
    output_filename = create_output_filename(input_filename, algorithm, rank, steps)
    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')
    print('input_filename: ', input_filename)
    output_file.write('input_filename: ' + input_filename + '\n')

    # Read and resize the input image.
    image = Image.open(input_filename)
    image = image.resize((dimensions[0], dimensions[1]), Image.ANTIALIAS)
    Y = np.array(image) / 256
    print(Y)
    plt.imshow(Y)
    plt.show()

    print('Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')

    print('rank: ', rank)
    output_file.write('rank: ' + str(rank) + '\n')
    print('seed: ', seed)
    output_file.write('seed: ' + str(seed) + '\n')
    print('algorithm: ', algorithm)
    output_file.write('algorithm: ' + str(algorithm) + '\n')
    print('l2_regularization: ', l2_regularization)
    output_file.write('l2_regularization: ' + str(l2_regularization) + '\n')
    print('steps: ', steps)
    output_file.write('steps: ' + str(steps) + '\n')
    print('epsilon: ', epsilon)
    output_file.write('epsilon: ' + str(epsilon) + '\n')
    print('delta: ', delta)
    output_file.write('delta: ' + str(delta) + '\n')
    print('downsampling_ratio: ', downsampling_ratio)
    output_file.write('downsampling_ratio: ' + str(downsampling_ratio) + '\n')
    output_file.flush()

    X_tucker = random_tucker(Y.shape, rank, random_state=seed)
    if algorithm in ['ALS', 'ALS-RS']:
        os.system('g++-10 -O2 -std=c++11 row_sampling.cc -o row_sampling')
        run_alternating_least_squares(X_tucker, Y, l2_regularization, algorithm, steps, epsilon, delta,
                                      downsampling_ratio, True)

    X = tl.tucker_to_tensor(X_tucker)
    # print(X)
    plt.imshow(X)
    plt.show()


# ==============================================================================
# Video Experiments
# - Reads an video as a 4-way tensor (time, x, y, RGB channel), and
# ==============================================================================
def run_video_experiment():
    input_filename = 'data/video/walking_past_camera.mp4'

    dimensions = [2493, 1080, 1920, 3]
    rank = [4, 4, 4, 2]

    seed = 0
    l2_regularization = 0.001
    steps = 5
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS-RS'
    # algorithm = 'ALS'

    global output_file
    output_filename = create_output_filename(input_filename, algorithm, rank, steps)
    print(output_filename)
    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')
    print('input_filename: ', input_filename)
    output_file.write('input_filename: ' + input_filename + '\n')

    # Read and resize the input image.
    video = skvideo.io.vread(input_filename)

    print('Tucker ...')
    # core, factors = tucker(video, rank=rank, init='random', tol=1e-5)
    # print('core', core.shape)
    # print('factors', len(factors))

    Y = np.array(video) / 256
    print('Original Y.shape: ', Y.shape)
    Y = Y[0:100, :, :, :]
    print('New Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')

    print('rank: ', rank)
    output_file.write('rank: ' + str(rank) + '\n')
    print('seed: ', seed)
    output_file.write('seed: ' + str(seed) + '\n')
    print('algorithm: ', algorithm)
    output_file.write('algorithm: ' + str(algorithm) + '\n')
    print('l2_regularization: ', l2_regularization)
    output_file.write('l2_regularization: ' + str(l2_regularization) + '\n')
    print('steps: ', steps)
    output_file.write('steps: ' + str(steps) + '\n')
    print('epsilon: ', epsilon)
    output_file.write('epsilon: ' + str(epsilon) + '\n')
    print('delta: ', delta)
    output_file.write('delta: ' + str(delta) + '\n')
    print('downsampling_ratio: ', downsampling_ratio)
    output_file.write('downsampling_ratio: ' + str(downsampling_ratio) + '\n')
    output_file.flush()

    X_tucker = random_tucker(Y.shape, rank, random_state=seed)
    if algorithm in ['ALS', 'ALS-RS']:
        os.system('g++-10 -O2 -std=c++11 row_sampling.cc -o row_sampling')
        run_alternating_least_squares(X_tucker, Y, l2_regularization, algorithm, steps, epsilon, delta,
                                      downsampling_ratio, True)

def main():
    # run_synthetic_experiment_1()
    # run_synthetic_shapes_experiment()
    # run_cardiac_mri_experiment()
    # run_image_experiment()
    run_video_experiment()


main()
