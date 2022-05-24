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
import dataclasses
from scipy import linalg

# TODO(fahrbach): Need to ensure that no new fields are added somehow.
# Note: If we add fields here, we need to update ReadAlgorithmConfig() protocol.
@dataclasses.dataclass
class AlgorithmConfig:
    # Instance info that defines optimization problem.
    input_shape: list[int] = None
    rank: list[int] = None
    l2_regularization_strength: float = 0.0

    # Algorithm parameters:
    # Expected to be in ['ALS', 'ALS-RS-Richardson', 'ALS-DJSSW19', 'ALS-naive', 'HOOI'].
    algorithm: str = 'ALS' 
    random_seed: int = 0

    # Parameters specific to 'ALS-RS'
    epsilon: float = 0.1
    delta: float = 0.01
    downsampling_ratio: float = 0.01 # 1.0 # 0.001
    max_num_samples: int = 0  # Optional to specify fixed number of samples.

    # Loop termination criteria.
    max_num_steps: int = 20
    rre_gap_tol: float = 1e-6  # Tracks relative residual errors of outer loop.

    # Logging info.
    verbose: bool = True  # Prints solve stats and info to STDOUT.

    # TODO(fahrbach): Add Verify() method.

# Returns (A_1 \ktimes ... \ktimes A_N) @ B
# TODO(fahrbach): Factor out to math_utils.py
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
    start_time = time.process_time()
    Y_matrix = tl.unfold(Y_tensor, factor_index)
    core_matrix = tl.unfold(X_tucker.core, factor_index)

    # Efficient computation of response matrix (i.e., all response vectors).
    start_time = time.process_time()
    factors = [X_tucker.factors[i].T for i in range(X_tucker.core.ndim) if i != factor_index]
    tmp = kron_mat_mult(factors, Y_matrix.T)
    response_matrix = tmp.T @ core_matrix.T
    if verbose:
        #print(' - KtB shape:', response_matrix.shape)
        print(' - KtB construction time:', time.process_time() - start_time)
        print(' - num least squares solves:',
                X_tucker.factors[factor_index].shape[0])
        #print(' - solve size:', AtA_lambda.shape, response_matrix.shape[1])

    # Efficient computation of normal matrix A^T A + \lambda * I.
    if l2_regularization < 0:       # Might have a bug atm?
        kron_invs = [X_tucker_factors_gram[i] for i in range(X_tucker.core.ndim) if i != factor_index]
        AtA = core_matrix @ kron_mat_mult(kron_invs, core_matrix.T)
        AtA_inv = np.linalg.pinv(AtA)
        ans = AtA_inv @ response_matrix.T
        X_tucker.factors[factor_index] = ans.T
    else:
        kron_squares = [X_tucker_factors_gram[i] for i in
                range(X_tucker.core.ndim) if i != factor_index]
        AtA_lambda = core_matrix @ kron_mat_mult(kron_squares, core_matrix.T) \
            + l2_regularization * np.identity(core_matrix.shape[0])
        if verbose:
            #print(' - KtK_lambda shape:', AtA_lambda.shape)
            print(' - KtK_lambda construction time:', time.process_time() - start_time)
    
        start_time = time.process_time()
        for row_index in range(X_tucker.factors[factor_index].shape[0]):
            X_tucker.factors[factor_index][row_index, :] = \
                np.linalg.solve(AtA_lambda, response_matrix[row_index, :])
        if verbose:
            print(' - total np.linalg.solve() time:',
                    time.process_time() - start_time)

    # Update Gram matrix of the new factor matrix.
    start_time = time.process_time()
    X_tucker_factors_gram[factor_index] = X_tucker.factors[factor_index].T @ \
        X_tucker.factors[factor_index]
    if verbose:
        print(' - X_tucker factor gram update time:', time.process_time() - start_time)

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
    Y_vec = tl.tensor_to_vec(Y_tensor)
    start_time = time.process_time()
    b = kron_mat_mult([factor.T for factor in X_tucker.factors], Y_vec)
    if verbose:
        print(' - Ktb construction time:', time.process_time() - start_time)
        #print(' - solve size:', KtK_lambda.shape, b.shape[0])

    if l2_regularization == 0.0:
        start_time = time.process_time()
        inv_grams = [np.linalg.pinv(X_tucker_factors_gram[n]) for n in range(len(X_tucker.factors))]
        new_core_vec = kron_mat_mult(inv_grams, b)
        X_tucker.core = tl.reshape(new_core_vec, X_tucker.core.shape)
        if verbose:
            print(' - kron_mat_mul core update time:', time.process_time() - start_time)
    else:
        start_time = time.process_time()
        KtK_lambda = np.identity(1)
        for n in range(len(X_tucker.factors)):
            KtK_lambda = np.kron(KtK_lambda, X_tucker_factors_gram[n])
        KtK_lambda += l2_regularization * np.identity(KtK_lambda.shape[0])
        if verbose:
            print(' - KtK_lambda construction time:', time.process_time() - start_time)

        start_time = time.process_time()
        new_core_tensor_vec = np.linalg.solve(KtK_lambda, b)
        X_tucker.core = tl.reshape(new_core_tensor_vec, X_tucker.core.shape)
        if verbose:
            print(' - np.linalg.solve() time:', time.process_time() - start_time)

# TODO(fahrbach): Can probably speed this up.
def compute_ridge_leverage_scores(A, l2_regularization):
    normal_matrix = A.T @ A + l2_regularization * np.identity(A.shape[1])
    normal_matrix_pinv = np.linalg.pinv(normal_matrix)
    leverage_scores = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        leverage_scores[i] = A[i, :] @ normal_matrix_pinv @ A[i, :].T
    return leverage_scores

# Sampling + Richardson iteration
def update_core_tensor_by_row_sampling(X_tucker, X_tucker_factors_gram,
        Y_tensor, config, step, output_file):
    l2_regularization = config.l2_regularization_strength
    epsilon = config.epsilon
    delta = config.delta
    downsampling_ratio = config.downsampling_ratio
    debug_mode = config.verbose
    
    d = 1
    for n in range(X_tucker.core.ndim):
        d *= config.rank[n]

    # Compute approximate ridge leverage scores for each factor matrix.
    start_time = time.process_time()
    leverage_scores = [compute_ridge_leverage_scores(factor, 0.0) for factor in
            X_tucker.factors]
    if debug_mode:
        print(' - leverage score computation time:', time.process_time() - start_time)
        
    num_samples = int(downsampling_ratio * 1680 * d * np.log(40 * d) * np.log(1.0 / delta) / epsilon)
    if config.max_num_samples > 0:
        num_samples = min(num_samples, config.max_num_samples)
    if debug_mode:
        print('num_samples:', num_samples)
    
    start_time = time.process_time()
    
    sampled_K = np.ones((1, num_samples))
    sampled_row_probability = np.ones(num_samples)
    sampled_row_indices_all = []
    for n in range(X_tucker.core.ndim):
        sum_ls = np.sum(leverage_scores[n])
        sampled_row_indices = np.random.choice(range(config.input_shape[n]),
                size=num_samples, p=list(leverage_scores[n] / sum_ls))
        sampled_row_probability = np.multiply(sampled_row_probability,
                leverage_scores[n][sampled_row_indices] / sum_ls)
        sampled_K = linalg.khatri_rao(sampled_K, X_tucker.factors[n][sampled_row_indices,:].T)
        sampled_row_indices_all.append(sampled_row_indices)
    sampled_K = sampled_K.T
    if debug_mode:
        print('sampled_K:', sampled_K.shape)
        print(' - constructing sampled K time:', time.process_time() - start_time)

    rescaling_coefficients = np.sqrt(np.ones(num_samples) / (num_samples * sampled_row_probability))
    del sampled_row_probability

    SK = np.einsum('i,ij->ij', rescaling_coefficients, sampled_K)
    del sampled_K

    # Convert from tensor index notation to vectorized indices.
    Sb = rescaling_coefficients * tl.tensor_to_vec(Y_tensor)[np.ravel_multi_index(sampled_row_indices_all, Y_tensor.shape)]

    KtStSb = SK.T @ Sb
    KtStSb = np.reshape(KtStSb, (len(KtStSb), 1))
    del Sb

    ######## Start Richardson iterations. ##########

    # Compute decomposition of M^+
    grams_Sigma = []
    grams_U = []
    for factor in X_tucker.factors:
        u, s, vt = np.linalg.svd(factor.T @ factor, full_matrices=True)
        grams_U.append(u)
        grams_Sigma.append(s)
    # Constuct diagonal matrix (as a vector)
    D = np.ones(1)
    for Sigma in grams_Sigma:
        D = np.kron(D, Sigma)
    D += (l2_regularization + 1e-6) * np.ones(D.shape[0])
    D = 1.0 / D
    D = np.reshape(D, (len(D), 1))
    del grams_Sigma

    x = np.zeros(d)
    x = np.reshape(x, (len(x), 1))

    for t in range(100):
        y = SK.T @ (SK @ x) + (l2_regularization * x) - KtStSb
        tmp = kron_mat_mult([U.T for U in grams_U], y)
        tmp = D * tmp
        tmp = kron_mat_mult(grams_U, tmp)
        z = x - (1 - epsilon**0.5) * tmp
        if t == 0:
            rre = 10**10
        else:
            rre = np.linalg.norm(z - x) / np.linalg.norm(x)
        x = z
        #print('step', t, 'rre', rre)
        if rre < 1e-6:
            break
    X_tucker.core = tl.reshape(x, X_tucker.core.shape)

def update_core_tensor_with_DJSSW19(X_tucker, Y_tensor, config, step, output_file):
    l2_regularization = config.l2_regularization_strength
    epsilon = config.epsilon
    delta = config.delta
    downsampling_ratio = config.downsampling_ratio
    debug_mode = config.verbose
    
    d = 1
    for n in range(X_tucker.core.ndim):
        d *= config.rank[n]

    # Compute approximate ridge leverage scores for each factor matrix.
    start_time = time.process_time()
    leverage_scores = [compute_ridge_leverage_scores(factor, 0.0) for factor in
            X_tucker.factors]
    if debug_mode:
        print(' - leverage score computation time:', time.process_time() - start_time)
        
    num_samples = int(downsampling_ratio * 1680 * d * np.log(40 * d) * np.log(1.0 / delta) / epsilon)
    if config.max_num_samples > 0:
        num_samples = min(num_samples, config.max_num_samples)
    if debug_mode:
        print('num_samples:', num_samples)
    
    start_time = time.process_time()
    
    sampled_K = np.ones((1, num_samples))
    sampled_row_probability = np.ones(num_samples)
    sampled_row_indices_all = []
    for n in range(X_tucker.core.ndim):
        sum_ls = np.sum(leverage_scores[n])
        sampled_row_indices = np.random.choice(range(config.input_shape[n]),
                size=num_samples, p=list(leverage_scores[n] / sum_ls))
        sampled_row_probability = np.multiply(sampled_row_probability,
                leverage_scores[n][sampled_row_indices] / sum_ls)
        sampled_K = linalg.khatri_rao(sampled_K, X_tucker.factors[n][sampled_row_indices,:].T)
        sampled_row_indices_all.append(sampled_row_indices)
    sampled_K = sampled_K.T
    if debug_mode:
        print('sampled_K:', sampled_K.shape)
        print(' - constructing sampled K time:', time.process_time() - start_time)

    rescaling_coefficients = np.sqrt(np.ones(num_samples) / (num_samples * sampled_row_probability))
    del sampled_row_probability

    SK = np.einsum('i,ij->ij', rescaling_coefficients, sampled_K)
    del sampled_K

    # Convert from tensor index notation to vectorized indices.
    Sb = rescaling_coefficients * tl.tensor_to_vec(Y_tensor)[np.ravel_multi_index(sampled_row_indices_all, Y_tensor.shape)]

    assert(config.l2_regularization_strength == 0.0)
    new_core_vec = np.linalg.pinv(SK) @ Sb
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

def RunTensorlyHooi(Y, config, output_file):
    assert(config.l2_regularization_strength == 0.0)
    start_time = time.process_time()
    core, tucker_factors = tucker(Y, rank=config.rank,
            n_iter_max=config.max_num_steps, init='random',
            tol=config.rre_gap_tol, verbose=True)
    end_time = time.process_time()
    print('total time:', end_time - start_time)
    if output_file:
        output_file.write('total time: ' + str(end_time - start_time) + '\n')
    print('avg step time:', (end_time - start_time) / config.max_num_steps)
    if output_file:
        output_file.write('HOOI avg time: ' + str((end_time - start_time) / config.max_num_steps) + '\n')

    
    X_tucker = random_tucker(Y.shape, config.rank, random_state=config.random_seed)
    X_tucker.factors = tucker_factors
    X_tucker.core = core
    loss = ComputeLossTerms(X_tucker, Y, config.l2_regularization_strength,
            np.linalg.norm(Y), Y.size)
    print('rre:', loss.rre)
    if output_file:
        output_file.write('rre: ' + str(loss.rre) + '\n')

def tucker_als(Y_tensor, config, output_file=None, X_tucker=None):
    # Separate case for running HOOI in Tensorly.
    if config.algorithm == 'HOOI':
        RunTensorlyHooi(Y_tensor, config, output_file)
        return

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
        if output_file:
            output_file.write('step: ' + str(step) + '\n')

        for factor_index in range(X_tucker.core.ndim):
            if config.verbose:
                print('Updating factor matrix:', factor_index)
            start_time = time.process_time()
            update_factor_matrix(X_tucker, X_tucker_factors_gram, Y_tensor,
                factor_index, config.l2_regularization_strength, config.verbose)
            end_time = time.process_time()

            loss_terms = ComputeLossTerms(X_tucker, Y_tensor,
                    config.l2_regularization_strength, Y_norm, num_elements)
            print('loss: {} rmse: {} rre: {} time: {}'.format(loss_terms.loss,
                loss_terms.rmse, loss_terms.rre, end_time - start_time))
            if output_file:
                output_file.write('loss: {} rmse: {} rre: {} time: {}'.format(
                    loss_terms.loss, loss_terms.rmse, loss_terms.rre, end_time - start_time) + '\n')
            if loss_terms.loss > all_loss_terms[-1].loss:
                print('Warning: The loss function increased!')
                if output_file:
                    output_file.write('Warning: The loss function increased!\n')
            all_loss_terms.append(loss_terms)

        if config.verbose:
            print('Updating core tensor:')
        start_time = time.process_time()
        if config.algorithm == 'ALS':
            update_core_tensor_memory_efficient(X_tucker,
                    X_tucker_factors_gram, Y_tensor,
                    config.l2_regularization_strength, config.verbose)
        elif config.algorithm == 'ALS-RS':
            update_core_tensor_by_row_sampling(X_tucker, X_tucker_factors_gram,
                    Y_tensor, config, step, output_file)
        elif config.algorithm == 'ALS-DJSSW19':
            update_core_tensor_with_DJSSW19(X_tucker, Y_tensor, config, step,
                    output_file)

        # Old implementations.
        elif config.algorithm == 'ALS-naive':
            update_core_tensor_naive(X_tucker, Y_tensor,
                    config.l2_regularization_strength)
        else:
            print('algorithm:', config.algorithm, 'is unsupported!')
            assert(False)
        end_time = time.process_time()

        loss_terms = ComputeLossTerms(X_tucker, Y_tensor,
                config.l2_regularization_strength, Y_norm, num_elements)
        print('loss: {} rmse: {} rre: {} time: {}'.format(loss_terms.loss,
            loss_terms.rmse, loss_terms.rre, end_time - start_time))
        if output_file:
            output_file.write('loss: {} rmse: {} rre: {} time: {}'.format(
                loss_terms.loss, loss_terms.rmse, loss_terms.rre, end_time - start_time) + '\n')
        if loss_terms.loss > all_loss_terms[-1].loss:
            print('Warning: The loss function increased!')
            if output_file:
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
