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
import dataclasses

# TODO(fahrbach): Need to ensure that no new fields are added somehow.
@dataclasses.dataclass
class AlgorithmConfig:
    # Instance info that defines optimization problem.
    input_shape: list[int] = None
    rank: list[int] = None
    l2_regularization_strength: float = 0.0

    # Algorithm parameters.
    algorithm: str = 'ALS' # Expected to be in ['ALS', 'ALS-RS', 'ALS-naive'].
    random_seed: int = 0

    # Parameters specific to 'ALS-RS'
    epsilon: float = 0.1
    delta: float = 0.1
    downsampling_ratio: float = 0.001

    # Loop termination criteria.
    max_num_steps: int = 20
    rre_gap_tol: float = 1e-6  # Tracks relative residual errors of outer loop.

    # Logging info.
    verbose: bool = True  # Prints solve stats and info to STDOUT.

    # TODO(fahrbach): Add Verify() method.

# TODO(fahrbach): Factor out to math_utils.py
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

# Writes vectorized version of leverage score vectors, factor matrices, and
# core tensors to temporary files. The shapes of these np.ndarrays can be
# reconstructed from the config file.
def write_regression_instance_to_files(config, leverage_scores, X_tucker, step):
    os.makedirs(os.path.dirname('tmp/'), exist_ok=True)

    with open('tmp/config.txt', 'w') as f:
        config_dict = dataclasses.asdict(config)
        for key in config_dict:
            f.write(str(key) + ' ' + str(config_dict[key]) + '\n')
        f.write('step ' + str(step) + '\n')
    for n in range(X_tucker.core.ndim):
        np.savetxt(f'tmp/leverage_scores_{n}_vec.txt',
                tl.tensor_to_vec(leverage_scores[n]))
        np.savetxt(f'tmp/factor_matrix_{n}_vec.txt',
                tl.tensor_to_vec(X_tucker.factors[n]))
    np.savetxt('tmp/core_tensor_vec.txt', tl.tensor_to_vec(X_tucker.core))

def update_core_tensor_by_row_sampling(X_tucker, Y_tensor, config, step,
        output_file):
    l2_regularization = config.l2_regularization_strength
    epsilon = config.epsilon
    delta = config.delta
    downsampling_ratio = config.downsampling_ratio
    debug_mode = config.verbose

    # Compute approximate ridge leverage scores for each factor matrix.
    start_time = time.time()
    leverage_scores = [compute_ridge_leverage_scores(factor, 0.0) for factor in
            X_tucker.factors]
    if debug_mode:
        print(' - leverage score computation time:', time.time() - start_time)

    # Write factor matrices, leverage score estimates, and core to `./tmp/`.
    start_time = time.time()
    write_regression_instance_to_files(config, leverage_scores, X_tucker, step)
    if debug_mode:
        print(' - write regression instance time:', time.time() - start_time)
    assert(False)

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

@dataclasses.dataclass
class LossTerms:
    residual_norm: float = None
    core_tensor_norm: float = None
    factor_matrix_norms: list[float] = None

    l2_regularization_strength: float = None

    loss: float = None
    rmse: float = None
    rre: float = None  # Relative residual error (ignores L2 regularization)

def ComputeLossTerms(X_tucker, Y_tensor, l2_regularization, Y_norm, Y_size):
    loss_terms = LossTerms()
    loss_terms.residual_norm = np.linalg.norm(Y_tensor -
            tl.tucker_to_tensor(X_tucker))
    loss_terms.core_tensor_norm = np.linalg.norm(X_tucker.core)
    loss_terms.factor_matrix_norms = [np.linalg.norm(f_matrix) for f_matrix in
            X_tucker.factors]

    loss_terms.l2_regularization_strength = l2_regularization

    loss_terms.loss = loss_terms.residual_norm**2
    loss_terms.loss += l2_regularization * loss_terms.core_tensor_norm**2
    for norm in loss_terms.factor_matrix_norms:
        loss_terms.loss += l2_regularization * norm**2
    loss_terms.rmse = np.sqrt(loss_terms.loss / Y_size)
    loss_terms.rre = loss_terms.residual_norm / Y_norm
    return loss_terms

def tucker_als(Y_tensor, config, output_file=None, X_tucker=None):
    if X_tucker == None:
        X_tucker = random_tucker(Y_tensor.shape, config.rank,
                random_state=config.random_seed)
    # Maintain the Gram matrix of each factor matrix.
    X_tucker_factors_gram = [X_tucker.factors[n].T @ X_tucker.factors[n] for n
            in range(X_tucker.core.ndim)]

    # Precompute quantities for faster loss computations.
    num_elements = Y_tensor.size
    Y_norm = np.linalg.norm(Y_tensor)

    loss_terms = ComputeLossTerms(X_tucker, Y_tensor,
            config.l2_regularization_strength, Y_norm, num_elements)
    all_loss_terms = [loss_terms]
    prev_outerloop_rre = loss_terms.rre

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

            loss_terms = ComputeLossTerms(X_tucker, Y_tensor,
                    config.l2_regularization_strength, Y_norm, num_elements)
            print('loss: {} rmse: {} rre: {} time: {}'.format(loss_terms.loss,
                loss_terms.rmse, loss_terms.rre, end_time - start_time))
            output_file.write('loss: {} rmse: {} rre: {} time: {}'.format(
                loss_terms.loss, loss_terms.rmse, loss_terms.rre, end_time - start_time) + '\n')
            if loss_terms.loss > all_loss_terms[-1].loss:
                print('Warning: The loss function increased!')
                output_file.write('Warning: The loss function increased!\n')
            all_loss_terms.append(loss_terms)

        if config.verbose:
            print('Updating core tensor:')
        start_time = time.time()
        if config.algorithm == 'ALS':
            update_core_tensor_memory_efficient(X_tucker,
                    X_tucker_factors_gram, Y_tensor,
                    config.l2_regularization_strength, config.verbose)
        elif config.algorithm == 'ALS-naive':
            update_core_tensor_naive(X_tucker, Y_tensor,
                    config.l2_regularization_strength)
        elif config.algorithm == 'ALS-RS':
            update_core_tensor_by_row_sampling(X_tucker, Y_tensor, config,
                    step, output_file)
        else:
            print('algorithm:', config.algorithm, 'is unsupported!')
            assert(False)
        end_time = time.time()

        loss_terms = ComputeLossTerms(X_tucker, Y_tensor,
                config.l2_regularization_strength, Y_norm, num_elements)
        print('loss: {} rmse: {} rre: {} time: {}'.format(loss_terms.loss,
            loss_terms.rmse, loss_terms.rre, end_time - start_time))
        output_file.write('loss: {} rmse: {} rre: {} time: {}'.format(
            loss_terms.loss, loss_terms.rmse, loss_terms.rre, end_time - start_time) + '\n')
        if loss_terms.loss > all_loss_terms[-1].loss:
            print('Warning: The loss function increased!')
            output_file.write('Warning: The loss function increased!\n')
        all_loss_terms.append(loss_terms)

        # Check relative residual error gap for early termination.
        rre_diff = prev_outerloop_rre - loss_terms.rre
        print('rre_diff:', rre_diff)
        if config.rre_gap_tol != None:
            if rre_diff >= 0 and rre_diff < config.rre_gap_tol:
                break
        prev_outerloop_rre = loss_terms.rre
        print()

    return X_tucker
