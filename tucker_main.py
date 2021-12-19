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
from dataclasses import dataclass

from tensor_data_handler import TensorDataHandler

# Global variable since all ALS subroutines log to this file.
output_file = None

# Tensor utils -----------------------------------------------------------------

def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)


def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)

# Algorithm config -------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    # Instance info that defines optimization problem.
    input_shape: list[int] = None
    rank: list[int] = None
    l2_regularization_strength: float = 0.0

    # Algorithm parameters.
    algorithm: str = 'ALS' # Expected to be in ['ALS', 'ALS-RS'].
    random_seed: int = 0

    # Parameters specific to 'ALS-RS'
    naive_core_update: bool = False  # Explicitly construct design matrix.

    # Parameters specific to 'ALS-RS'
    epsilon: float = 0.1
    delta: float = 0.1
    downsampling_ratio: float = 0.0001

    # Loop termination criteria.
    max_num_steps: int = 20
    rre_gap: float = 1e-6  # Tracks relative residual errors of outer loop.

    # Logging info.
    verbose: bool = True  # Prints solve stats and info to STDOUT.

# TODO(fahrbach): Describe runtime. Returns (A_1 \ktimes ... \ktimes A_N) @ B
def kron_mat_mult(kroneckor_matrices, matrix):
    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (len(matrix), 1))
    mat_shape = np.shape(matrix)
    num_mats = len(kroneckor_matrices)
    num_rows_kron_mats = np.zeros((num_mats), dtype=np.int32)
    num_cols_kron_mats = np.zeros((num_mats), dtype=np.int32)
    for i in range(num_mats):
        num_rows_kron_mats[i] = np.shape(kroneckor_matrices[i])[0]
        num_cols_kron_mats[i] = np.shape(kroneckor_matrices[i])[1]
    vec_size = np.prod(num_cols_kron_mats)
    if vec_size != mat_shape[0]:
        raise ValueError(
            'The number of columns of the Kronecker product should match ' +
            'the number of rows in the matrix.')
    output = matrix
    for j in range(num_mats - 1, -1, -1):
        output = np.reshape(output, (num_cols_kron_mats[j], \
                mat_shape[1] * vec_size // num_cols_kron_mats[j]), 'F')
        output = np.matmul(kroneckor_matrices[j], output)
        vec_size = vec_size * num_rows_kron_mats[j] // num_cols_kron_mats[j]
        output = np.reshape(output, (num_rows_kron_mats[j], mat_shape[1],
            vec_size // num_rows_kron_mats[j]))
        output = np.transpose(
            np.reshape(np.moveaxis(output, [0, 1, 2], [1, 0, 2]),
                (mat_shape[1], vec_size)))
    return output

# TODO(fahrbach): Show why higher-order SVD methods do not scale well? Seems
# that they might if we use compressed representation?
# Note: `X_tucker_factors_gram` must be a list of the gram matrices for the
# current factor matrices (i.e., X_tucker.factors[i].T @ X_tucker.factors[i]).
# - Matricized version of Eq 4.2 in "Tensor Decompositions and Applications."
def update_factor_matrix(X_tucker, X_tucker_factors_gram, Y_tensor,
        factor_index, l2_regularization, verbose):
    start_time = time.time()
    Y_matrix = tl.unfold(Y_tensor, factor_index)
    core_matrix = tl.unfold(X_tucker.core, factor_index)

    # Efficient computation of normal matrix A^T A + \lambda * I.
    kron_squares = [X_tucker_factors_gram[i] for i in
            range(X_tucker.core.ndim) if i != factor_index]
    AtA_lambda = core_matrix @ kron_mat_mult(kron_squares, core_matrix.T) \
        + l2_regularization * np.identity(core_matrix.shape[0])
    if verbose:
        #print(' - KtK_lambda shape:', AtA_lambda.shape)
        print(' - KtK_lambda construction time:', time.time() - start_time)

    # Efficient computation of response matrix (i.e., all response vectors).
    start_time = time.time()
    factors = [X_tucker.factors[i].T for i in range(X_tucker.core.ndim) if i != factor_index]
    tmp = kron_mat_mult(factors, Y_matrix.T)
    response_matrix = tmp.T @ core_matrix.T
    if verbose:
        #print(' - KtB shape:', response_matrix.shape)
        print(' - KtB construction time:', time.time() - start_time)
        print(' - num least squares solves:',
                X_tucker.factors[factor_index].shape[0])
        print(' - solve size:', AtA_lambda.shape, response_matrix.shape[1])
    
    start_time = time.time()
    for row_index in range(X_tucker.factors[factor_index].shape[0]):
        X_tucker.factors[factor_index][row_index, :] = \
            np.linalg.solve(AtA_lambda, response_matrix[row_index, :])
    if verbose:
        print(' - total np.linalg.solve() time:',
                time.time() - start_time)

    # Update Gram matrix of the new factor matrix.
    start_time = time.time()
    X_tucker_factors_gram[factor_index] = X_tucker.factors[factor_index].T @ \
        X_tucker.factors[factor_index]
    if verbose:
        print(' - X_tucker factor gram update time:', time.time() - start_time)

# Naive core tensor update that explicitly constructs the design matrix. This
# requires O((I_1 * I_2 * I_3) * (R_1 * R_2 * R_3)) space, and is prohibitively
# expensive for anything interesting. It seems to be less accurate numerically?
def update_core_tensor_naive(X_tucker, Y_tensor, l2_regularization):
    design_matrix = np.identity(1)
    for n in range(X_tucker.core.ndim):
        design_matrix = np.kron(design_matrix, X_tucker.factors[n])

    Y_vec = tl.tensor_to_vec(Y_tensor)

    AtA_lambda = design_matrix.T @ design_matrix
    AtA_lambda += l2_regularization * np.identity(design_matrix.shape[1])
    Atb = design_matrix.T @ Y_vec
    X_tucker_core = tl.reshape(np.linalg.solve(AtA_lambda, Atb),
            X_tucker.core.shape)

# Memory-efficient construction of the normal equation for core tensor update.
def update_core_tensor_memory_efficient(X_tucker, X_tucker_factors_gram,
        Y_tensor, l2_regularization, verbose):
    start_time = time.time()
    KtK_lambda = np.identity(1)
    for n in range(len(X_tucker.factors)):
        KtK_lambda = np.kron(KtK_lambda, X_tucker_factors_gram[n])
    KtK_lambda += l2_regularization * np.identity(KtK_lambda.shape[0])
    if verbose:
        print(' - KtK_lambda construction time:', time.time() - start_time)

    Y_vec = tl.tensor_to_vec(Y_tensor)
    start_time = time.time()
    b = kron_mat_mult([factor.T for factor in X_tucker.factors], Y_vec)
    if verbose:
        print(' - Ktb construction time:', time.time() - start_time)
        print(' - solve size:', KtK_lambda.shape, b.shape[0])

    start_time = time.time()
    new_core_tensor_vec = np.linalg.solve(KtK_lambda, b)
    X_tucker.core = tl.reshape(new_core_tensor_vec, X_tucker.core.shape)
    if verbose:
        print(' - np.linalg.solve() time:', time.time() - start_time)


def compute_ridge_leverage_scores(A, l2_regularization):
    normal_matrix = A.T @ A + l2_regularization * np.identity(A.shape[1])
    normal_matrix_pinv = np.linalg.pinv(normal_matrix)
    leverage_scores = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        leverage_scores[i] = A[i, :] @ normal_matrix_pinv @ A[i, :].T
    return leverage_scores


def write_leverage_scores_to_file(leverage_scores, X_tucker, l2_regularization,
        epsilon, delta, step, alpha):
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


def update_core_tensor_by_row_sampling(X_tucker, Y_tensor, l2_regularization,
        step, epsilon, delta, downsampling_ratio, debug_mode):
    global output_file

    # Compute approximate ridge leverage scores for each factor matrix.
    start_time = time.time()
    leverage_scores = [compute_ridge_leverage_scores(factor, 0.0) for factor in X_tucker.factors]
    if debug_mode:
        print(' - leverage score computation time:', time.time() - start_time)

    # Write factor matrices, leverage score estimates, and core to `./tmp/`.
    start_time = time.time()
    write_leverage_scores_to_file(leverage_scores, X_tucker, l2_regularization,
            epsilon, delta, step, downsampling_ratio)
    cmd = './row_sampling'
    os.system(cmd)
    if debug_mode:
        print(' - row sampling subroutine time:', time.time() - start_time)

    num_original_rows = 1
    num_augmented_rows = 1
    for n in range(X_tucker.core.ndim):
        num_original_rows *= X_tucker.factors[n].shape[0]
        num_augmented_rows *= X_tucker.factors[n].shape[1]

    num_core_elements = 1
    for dimension in X_tucker.core.shape:
        num_core_elements *= dimension

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
            if line_index % 100 == 0:
                print(line_index, line_index / len(lines))
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

# Relative residual error.
def compute_relative_residual_error(Y_tensor, X_tucker):
    residual_vec = tl.tensor_to_vec(Y_tensor - tl.tucker_to_tensor(X_tucker))
    return np.linalg.norm(residual_vec) / np.linalg.norm(tl.tensor_to_vec(Y_tensor))


#def tucker_als(X_tucker, Y_tensor, l2_regularization, algorithm, num_steps,
#        epsilon, delta, downsampling_ratio, debug_mode):
# TODO(fahrbach): Describe this function.
def tucker_als(Y_tensor, config, output_file, X_tucker=None):
    # Initialize Tucker decomposition if not provided as input.
    if X_tucker == None:
        X_tucker = random_tucker(Y_tensor.shape, config.rank,
                random_state=config.random_seed)
    # Maintain the Gram matrix of each factor matrix.
    X_tucker_factors_gram = [X_tucker.factors[n].T @ X_tucker.factors[n] for n
            in range(X_tucker.core.ndim)]

    num_elements = np.prod(Y_tensor.shape)
    loss = compute_loss(Y_tensor, X_tucker, config.l2_regularization_strength)

    for step in range(config.max_num_steps):
        print('step:', step)
        output_file.write('step: ' + str(step) + '\n')

        for factor_index in range(X_tucker.core.ndim):
            if config.verbose:
                print('Updating factor matrix:', factor_index)
            start_time = time.time()
            update_factor_matrix(X_tucker, X_tucker_factors_gram, Y_tensor,
                factor_index, config.l2_regularization_strength, config.verbose)
            end_time = time.time()

            new_loss = compute_loss(Y_tensor, X_tucker, config.l2_regularization_strength)
            rmse = (new_loss / num_elements) ** 0.5
            rre = compute_relative_residual_error(Y_tensor, X_tucker)
            print('loss: {} RMSE: {} RRE: {} time: {}'.format(new_loss, rmse,
                rre, end_time - start_time))
            output_file.write('loss: {} RMSE: {} RRE: {} time: {}'.format(
                new_loss, rmse, rre, end_time - start_time) + '\n')
            if new_loss > loss:
                print('Warning: The loss function increased!')
                output_file.write('Warning: The loss function increased!\n')
            output_file.flush()
            loss = new_loss

        if config.verbose:
            print('Updating core tensor:')
        start_time = time.time()
        if config.algorithm == 'ALS':
            if config.naive_core_update:
                update_core_tensor_naive(X_tucker, Y_tensor,
                        config.l2_regularization_strength)
            else:
                update_core_tensor_memory_efficient(X_tucker,
                        X_tucker_factors_gram, Y_tensor,
                        config.l2_regularization_strength, config.verbose)
        elif config.algorithm == 'ALS-RS':
            update_core_tensor_by_row_sampling(X_tucker, Y_tensor,
                    l2_regularization, step, epsilon, delta,
                    downsampling_ratio, debug_mode)
        else:
            print('algorithm:', config.algorithm, 'is unsupported!')
            assert(False)
        end_time = time.time()

        new_loss = compute_loss(Y_tensor, X_tucker,
                config.l2_regularization_strength)
        rmse = (new_loss / num_elements) ** 0.5
        rre = compute_relative_residual_error(Y_tensor, X_tucker)
        print('loss: {} RMSE: {} RRE: {} time: {}'.format(new_loss, rmse, rre,
            end_time - start_time))
        output_file.write('loss: {} RMSE: {} RRE: {} time: {}'.format(new_loss,
            rmse, rre, end_time - start_time) + '\n')
        if new_loss > loss:
            print('Warning: The loss function increased!')
            output_file.write('Warning: The loss function increased!\n')
        output_file.flush()
        loss = new_loss
        print()
    return X_tucker

# Creates output filename based on input algorithm parameters, makes output path
# if it doesn't already exist, and initializes global `output_file` variable.
def init_output_file(output_filepath_prefix, algorithm, rank, steps):
    global output_file

    output_filename = output_filepath_prefix
    output_filename += '_' + algorithm
    output_filename += '_' + ','.join([str(x) for x in rank])
    output_filename += '_' + str(steps)
    output_filename += '.txt'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'a')

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
    shape = (50, 50, 100, 100)
    rank = (4, 4, 10, 4)
    steps = 10
    l2_regularization = 0.001
    seed = 0
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS'
    # algorithm = 'ALS-RS'

    data_handler = TensorDataHandler()
    data_handler.load_random_tucker(shape, [10, 10, 10, 10], random_state=(seed + 1000))

    global output_file
    output_filename = data_handler.output_filename_prefix
    output_filename += '_' + ','.join([str(x) for x in shape])
    output_filename += '_' + ','.join([str(x) for x in rank])
    output_filename += '_' + str(seed + 1000)
    output_filename += '_' + algorithm
    output_filename += '.txt'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'a')

    output_file.write('##############################################\n')

    # Initialize target tensor Y.
    Y = data_handler.tensor
    Y[(0, 0, 0, 0)] = 0

    print('Y.shape: ', Y.shape)
    output_file.write('Y.shape: ' + str(Y.shape) + '\n')
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
    tucker_als(X_tucker, Y, l2_regularization, algorithm, steps, epsilon,
            delta, downsampling_ratio, True)
    #X = tl.tucker_to_tensor(X_tucker)
    #print(X)

# ==============================================================================
# Synthetic Shapes Experiment:
# - Use Tensorly's built-in shape images.
# - Note: This data can easily scale up, and starts to show the benefit of
#   row sampling. For example, create a shape of dimensions [1024, 1024, 3]
#   and rank [4, 4, 3]. Observe that it's only sampling about 0.1% of the rows.
# ==============================================================================
def run_synthetic_shapes_experiment():
    pattern = 'circle'  # ['rectangle', 'swiss', 'circle']
    n = 100
    rank = [10, 10, 2]
    steps = 10
    l2_regularization = 0.001
    seed = 0
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0
    algorithm = 'ALS'
    # algorithm = 'ALS-RS'

    data_handler = TensorDataHandler()
    data_handler.load_synthetic_shape(pattern, n, n, 3)

    global output_file
    output_filename_prefix = data_handler.output_filename_prefix
    output_filename_prefix += '_' + str(n)
    init_output_file(output_filename_prefix, algorithm, rank, steps)

    output_file.write('##############################################\n')

    # Initialize target tensor Y.
    Y = data_handler.tensor
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

# ==============================================================================
# Cardiac MRI Experiment:
# - Read 4-way tensor with shape (256, 256, 14, 20), which corresponds to
#   positions (x, y, z, time), and run ALS with and without row sampling.
# ==============================================================================
def run_cardiac_mri_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_cardiac_mri_data()

    #algorithm = 'ALS-RS'
    algorithm = 'ALS'
    rank = (10, 10, 4, 4)
    seed = 0
    l2_regularization = 0.001
    steps = 10
    epsilon = 0.1
    delta = 0.1
    downsampling_ratio = 1.0

    Y = data_handler.tensor

    global output_file
    output_filename_prefix = data_handler.output_filename_prefix
    init_output_file(output_filename_prefix, algorithm, rank, steps)

    output_file.write('##############################################\n')
    print('input_filename: ', data_handler.input_filename)
    output_file.write('input_filename: ' + data_handler.input_filename + '\n')

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
    tucker_als(X_tucker, Y, l2_regularization, algorithm, steps, epsilon,
            delta, downsampling_ratio, True)

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# ==============================================================================
# Image Experiments
# - Reads an image as a 3-way tensor (x, y, RGB channel), and 
# ==============================================================================
def run_image_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_image('data/images/nyc.jpg', resize_shape=(500, 320))
    #data_handler.load_image('data/images/nyc.jpg', resize_shape=(2000, 1280))

    config = AlgorithmConfig()

    config.input_shape = data_handler.tensor.shape
    config.rank = [25, 25, 2]
    config.l2_regularization = 0.001

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS'
    #config.verbose = False
    print(config)

    global output_file
    # TODO(fahrbach): Create output_file from data_handler + config.
    init_output_file(data_handler.output_filename_prefix, config.algorithm,
            config.rank, config.max_num_steps)

    output_file.write('##############################################\n')
    # TODO(fahrbach): Write info from DataHandler to output_file
    print('input_filename: ', data_handler.input_filename)
    output_file.write('input_filename: ' + data_handler.input_filename + '\n')

    # TODO(fahrbach): Write AlgorithmConfig + instance stats (e.g., compression)
    # to output_file.
    # WriteConfigAndStatsToOutput()

    """
    # Compression factor
    tucker_size = 0
    core_size = 1
    for i in range(len(dimensions)):
        tucker_size += dimensions[i] * rank[i]
        core_size *= rank[i]
    tucker_size += core_size
    print('tucker_size:', tucker_size)
    output_file.write('tucker_size: ' + str(tucker_size) + '\n')
    compression_factor = Y.size / tucker_size
    print('compression_factor:', compression_factor)
    output_file.write('compression_factor: ' + str(compression_factor) + '\n')
    output_file.flush()
    """

    Y = data_handler.tensor
    #print(Y)
    #plt.imshow(Y)
    #plt.show()
    X_tucker = random_tucker(Y.shape, config.rank, random_state=config.random_seed)
    # Inputs: Y, algorithm_config, output_file, X_tucker=None
    X_tucker = tucker_als(Y, config, output_file, X_tucker=None)

    #tucker_als(X_tucker, Y, config.l2_regularization_strength, config.algorithm, config.max_num_steps, config.epsilon, config.delta, config.downsampling_ratio, config.verbose)

    # Plotting the original and reconstruction from the decompositions
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axis_off()
    ax.imshow(to_image(Y))
    ax.set_title('original')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_axis_off()
    ax.imshow(to_image(tl.tucker_to_tensor(X_tucker)))
    ax.set_title('Tucker')

    plt.tight_layout()
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
    init_output_file(input_filename, algorithm, rank, steps)

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
    #run_cardiac_mri_experiment()
    run_image_experiment()
    # run_video_experiment()


main()
