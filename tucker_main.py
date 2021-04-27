import numpy as np
import tensorly as tl
from tensorly.random import random_tucker
import time

# TODO(fahrbach): Use a logger instead of print statements.

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
def update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization):
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
    for row_index in range(X_tucker.factors[factor_index].shape[0]):
        response_vec = Y_matrix[row_index,:]
        X_tucker.factors[factor_index][row_index,:] = solve_least_squares( \
                design_matrix, response_vec, l2_regularization)

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
def update_core_tensor_memory_efficient(X_tucker, Y_tensor, l2_regularization):
    KtK_lambda = np.identity(1)
    for n in range(len(X_tucker.factors)):
        KtK_lambda = np.kron(KtK_lambda, \
                X_tucker.factors[n].T @ X_tucker.factors[n])
    KtK_lambda += l2_regularization * np.identity(KtK_lambda.shape[0])

    factors_T = [factor.T.copy() for factor in X_tucker.factors]
    Y_vec = tl.tensor_to_vec(Y_tensor)
    b = np.zeros(KtK_lambda.shape[0])
    row_index = 0

    # Note: fahrbach thinks this is still much slower than it could be due to
    # Python loop slowdowns. Is it possible to only use NumPY operations without
    # using too much memory?
    if X_tucker.core.ndim == 2:
        for core_index_0 in range(X_tucker.core.shape[0]):
            factor_matrix_col_T_0 = X_tucker.factors[0][:,core_index_0].T
            design_matrix_col_T_0 = factor_matrix_col_T_0
            for core_index_1 in range(X_tucker.core.shape[1]):
                factor_matrix_col_T_1 = X_tucker.factors[1][:,core_index_1].T
                design_matrix_col_T_1 = np.kron(design_matrix_col_T_0, factor_matrix_col_T_1)
                b[row_index] = design_matrix_col_T_1 @ Y_vec
                row_index += 1
    elif X_tucker.core.ndim == 3:
        for core_index_0 in range(X_tucker.core.shape[0]):
            factor_matrix_col_T_0 = X_tucker.factors[0][:,core_index_0].T
            design_matrix_col_T_0 = factor_matrix_col_T_0
            for core_index_1 in range(X_tucker.core.shape[1]):
                factor_matrix_col_T_1 = X_tucker.factors[1][:,core_index_1].T
                design_matrix_col_T_1 = np.kron(design_matrix_col_T_0, factor_matrix_col_T_1)
                for core_index_2 in range(X_tucker.core.shape[2]):
                    factor_matrix_col_T_2 = X_tucker.factors[2][:,core_index_2].T
                    design_matrix_col_T_2 = np.kron(design_matrix_col_T_1, factor_matrix_col_T_2)
                    b[row_index] = design_matrix_col_T_2 @ Y_vec
                    row_index += 1
    else:
        print('Core tensor of order', X_tucker.core.ndim, 'not supported.')
        assert(False)

    new_core_tensor_vec = np.linalg.solve(KtK_lambda, b)
    X_tucker.core = tl.reshape(new_core_tensor_vec, X_tucker.core.shape)

def compute_ridge_leverage_scores(A, l2_regularization=0.0):
    normal_matrix = A.T @ A + l2_regularization * np.identity(A.shape[1])
    normal_matrix_pinv = np.linalg.pinv(normal_matrix)
    leverage_scores = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        leverage_scores[i] = A[i,:] @ normal_matrix_pinv @ A[i,:].T
    return leverage_scores

def update_core_tensor_by_row_sampling(X_tucker, Y_tensor, l2_regularization):
    print('Starting to compute leverage scores.')
    leverage_scores = [compute_ridge_leverage_scores(factor) for factor in X_tucker.factors]
    print('Finished.')

    # This part of the core update is the bottleneck down due to cache misses.
    print('Start sampling')
    X_shape = []
    num_elements = 1
    for n in range(X_tucker.core.ndim):
        X_shape.append(X_tucker.factors[n].shape[0])
        num_elements *= X_shape[-1]

    # TODO(fahrbach): We need to do the sampling in C++ since Python for loops
    # are so slow. This is fair since all of NumPY is C or Fortran.
    row_index = 0
    while row_index < num_elements:
    #for row_index in range(num_elements):
        kronecker_index = vec_index_to_tensor_index(row_index, X_shape)
        leverage_score = 1.0
        for n in range(X_tucker.core.ndim):
            factor_row_index = kronecker_index[n]
            leverage_score *= leverage_scores[n][factor_row_index]
        #print(row_index, kronecker_index, leverage_score)
        row_index += 1
    print('Finished.')
        
def compute_loss(Y_tensor, X_tucker, l2_regularization):
    loss = 0.0
    residual_vec = tl.tensor_to_vec(Y_tensor - tl.tucker_to_tensor(X_tucker))
    loss += np.linalg.norm(residual_vec)**2
    loss += l2_regularization * np.linalg.norm(tl.tensor_to_vec(X_tucker.core))**2
    for n in range(len(X_tucker.factors)):
        loss += l2_regularization * np.linalg.norm(X_tucker.factors[n])**2
    return loss

# Simple tensor decomposition experiment that uses alternating least squares to
# decompose a tensor Y generated from a random Tucker decomposition.
def main():
    shape = (300, 400, 40)
    rank = (10, 50, 2)
    l2_regularization = 0.01
    order = len(shape)
    num_elements = 1
    for dimension in shape:
        num_elements *= dimension

    # --------------------------------------------------------------------------
    # Intialize tensors.
    Y_tucker = random_tucker(shape, rank, random_state=0)
    Y_tensor = tl.tucker_to_tensor(Y_tucker)
    Y_tensor[0,0,0] = 1
    print('Target tensor Y:')
    print(Y_tucker)
    print()

    X_tucker = random_tucker(shape, rank, random_state=1)
    print('Learned tensor X:')
    print(X_tucker)
    print()

    loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
    print('Loss:', loss)
    print()

    for step in range(10):
        print('Step:', step)
        # --------------------------------------------------------------------------
        # Factor matrix updates:
        for factor_index in range(X_tucker.core.ndim):
            #print('Updating factor matrix:', factor_index)
            start_time = time.time()
            update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization)
            end_time = time.time()

            new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
            rmse = (new_loss / num_elements)**0.5
            print('Loss: {} RMSE: {} Time: {}'.format(new_loss, rmse, end_time - start_time))
            if new_loss > loss:
                print('Warning: The loss function increased!')
            loss = new_loss
        
        # --------------------------------------------------------------------------
        # Core tensor update:
        start_time = time.time()
        update_core_tensor_memory_efficient(X_tucker, Y_tensor, l2_regularization)
        #update_core_tensor_by_row_sampling(X_tucker, Y_tensor, l2_regularization)
        end_time = time.time()

        new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
        rmse = (new_loss / num_elements)**0.5
        print('Loss: {} RMSE: {} Time: {}'.format(new_loss, rmse, end_time - start_time))
        if new_loss > loss:
            print('Warning: The loss function increased!')
        loss = new_loss

main()
