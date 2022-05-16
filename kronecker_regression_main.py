from tucker_als import *

import argparse
import datetime
import numpy as np
import time

def create_regression_experiment_01(num_dim, n, d, seed=0):
    np.random.seed(seed)
    mean = 1.0
    stdev = 0.001
    return [np.random.normal(mean, stdev, size=(n, d)) for i in range(num_dim)]

def kronecker_product(factors):
    tmp = np.ones((1, 1))
    for i in range(len(factors)):
        tmp = np.kron(tmp, factors[i])
    return tmp

def loss_function(factors, l2_regularization, x):
    if len(x.shape) == 1:
        x = np.reshape(x, (len(x), 1))
    num_rows = 1
    for factor in factors:
        num_rows *= factor.shape[0]
    r = kron_mat_mult(factors, x) - np.ones((num_rows, 1))
    loss = np.linalg.norm(r)**2
    loss += l2_regularization * np.linalg.norm(x)**2
    return loss

# Algorithm 01: Normal equations, take advantage of some Kronecker structure
def kronecker_regression_algorithm_01(factors, l2_regularization):
    start = time.process_time()
    K = kronecker_product(factors)
    print('K:', K.shape)
    n, d = K.shape
    normal_matrix = np.ones((1, 1))
    for i in range(len(factors)):
        normal_matrix = np.kron(normal_matrix, factors[i].T @ factors[i])
    assert normal_matrix.shape[0] == normal_matrix.shape[1]
    normal_matrix += l2_regularization**0.5 * np.identity(d)
    print('normal_matrix:', normal_matrix.shape)

    pinv_normal = np.linalg.pinv(normal_matrix)
    print('pinv_normal:', pinv_normal.shape)
        
    b = np.ones(n)
    x = pinv_normal @ (K.T @ b)
    print('time:', time.process_time() - start)
    print('loss:', loss_function(factors, l2_regularization, x))

# Algorithm 02: Normal equations, KronMatMul for computing K.T @ b.
def kronecker_regression_algorithm_02(factors, l2_regularization):
    start = time.process_time()
    K = kronecker_product(factors)
    print('K:', K.shape)
    n, d = K.shape
    normal_matrix = np.ones((1, 1))
    for i in range(len(factors)):
        normal_matrix = np.kron(normal_matrix, factors[i].T @ factors[i])
    assert normal_matrix.shape[0] == normal_matrix.shape[1]
    normal_matrix += l2_regularization**0.5 * np.identity(d)
    print('normal_matrix:', normal_matrix.shape)

    pinv_normal = np.linalg.pinv(normal_matrix)
    print('pinv_normal:', pinv_normal.shape)
        
    b = np.ones(n)
    Ktb = kron_mat_mult([A.T for A in factors], b)
    x = pinv_normal @ Ktb
    print('time:', time.process_time() - start)
    print('loss:', loss_function(factors, l2_regularization, x))

# Algorithm 03: Normal equations, SVD for each factors + KronMatMul
def kronecker_regression_algorithm_03(factors, l2_regularization):
    start = time.process_time()
    factors_Sigma = []
    factors_Vt = []
    for factor in factors:
        u, s, vt = np.linalg.svd(factor, full_matrices=True)
        factors_Sigma.append(s)
        factors_Vt.append(vt)
        #print(u.shape, s.shape, vt.shape)

    num_rows = 1
    for factor in factors:
        num_rows *= factor.shape[0]
    b = np.ones(num_rows)
    Ktb = kron_mat_mult([A.T for A in factors], b)
    print('Ktb:', Ktb.shape)
    # Multiply by SVD of normal matrix pseudoinverse from factor SVDs.
    tmp = kron_mat_mult(factors_Vt, Ktb)
    print('tmp:', tmp.shape)
    # Constuct diagonal matrix (as a vector)
    D = np.ones(1)
    for Sigma in factors_Sigma:
        D = np.kron(D, Sigma * Sigma)
    D += l2_regularization * np.ones(D.shape[0])
    D = 1.0 / D
    D = np.reshape(D, (len(D), 1))
    tmp = D * tmp
    x = kron_mat_mult([Vt.T for Vt in factors_Vt], tmp)
    print('time:', time.process_time() - start)
    print('loss:', loss_function(factors, l2_regularization, x))

# Algorithm 04: Row sampling K, pseudoinverse on (SK)^T (SK) -- Diao et al.
# - epsilon: approximation guarantee for least squares error
# - delta: failure probability
# - alpha: downsampling factor
def kronecker_regression_algorithm_04(factors, l2_regularization, epsilon, delta, alpha):
    start = time.process_time()

    ls = [compute_ridge_leverage_scores(factor, 0.0) for factor in factors] 
    print('leverage_scores:', [sum(p) for p in ls])
    num_columns = 1
    for factor in factors:
        num_columns *= factor.shape[1]
    print('num_columns:', num_columns)
    num_samples = int(alpha * 1680 * num_columns * np.log(40 * num_columns) * np.log(1.0 / delta) / epsilon)
    print('num_samples:', num_samples)

    # Row sampling code from core tensor update.
    sampled_K = np.ones((1, num_samples))  # Rows of K that are sampled before any rescaling.
    sampled_row_probability = np.ones(num_samples)
    for n in range(len(factors)):
        sum_ls = np.sum(ls[n])
        sampled_row_indices = np.random.choice(range(factors[n].shape[0]),
                size=num_samples, p=list(ls[n] / sum_ls))
        sampled_row_indices = np.sort(sampled_row_indices)
        print('samples:', n, len(sampled_row_indices), sampled_row_indices)
        sampled_row_probability = np.multiply(sampled_row_probability,
                ls[n][sampled_row_indices] / sum_ls)
        sampled_K = linalg.khatri_rao(sampled_K, factors[n][sampled_row_indices,:].T)
    sampled_K = sampled_K.T
    print('sampled_K:', sampled_K.shape)

    rescaling_coefficients = np.sqrt(np.ones(num_samples) / (num_samples * sampled_row_probability))
    print('rescaling_coefficients:', rescaling_coefficients.shape)

    SK = np.einsum('i,ij->ij', rescaling_coefficients, sampled_K)
    print('SK:', SK.shape)
    del sampled_K, sampled_row_probability

    response_vec = np.ones(SK.shape[0])
    Sb = rescaling_coefficients * response_vec
    print('Sb:', Sb.shape)
    del rescaling_coefficients

    SAtSA = SK.T @ SK + np.identity(num_columns) * l2_regularization
    print('normal matrix:', SAtSA.shape)

    tmp = SK.T @ Sb
    normal_pinv = np.linalg.pinv(SAtSA)
    x = normal_pinv @ tmp
    print('x:', x.shape)
    #print(x)

    print('time:', time.process_time() - start)
    print('loss:', loss_function(factors, l2_regularization, x))

# Algorithm 05: Row sampling K, Richardson iterations w/ KronMatMul preconditioner
# - epsilon: approximation guarantee for least squares error
# - delta: failure probability
# - alpha: downsampling factor
def kronecker_regression_algorithm_05(factors, l2_regularization, epsilon, delta, alpha):
    start = time.process_time()

    ls = [compute_ridge_leverage_scores(factor, 0.0) for factor in factors] 
    print('leverage_scores:', [sum(p) for p in ls])
    num_columns = 1
    for factor in factors:
        num_columns *= factor.shape[1]
    print('num_columns:', num_columns)
    num_samples = int(alpha * 1680 * num_columns * np.log(40 * num_columns) * np.log(1.0 / delta) / epsilon)
    print('num_samples:', num_samples)

    # Row sampling code from core tensor update.
    sampled_K = np.ones((1, num_samples))  # Rows of K that are sampled before any rescaling.
    sampled_row_probability = np.ones(num_samples)
    for n in range(len(factors)):
        sum_ls = np.sum(ls[n])
        sampled_row_indices = np.random.choice(range(factors[n].shape[0]),
                size=num_samples, p=list(ls[n] / sum_ls))
        sampled_row_indices = np.sort(sampled_row_indices)
        print('samples:', n, len(sampled_row_indices), sampled_row_indices)
        sampled_row_probability = np.multiply(sampled_row_probability,
                ls[n][sampled_row_indices] / sum_ls)
        sampled_K = linalg.khatri_rao(sampled_K, factors[n][sampled_row_indices,:].T)
    sampled_K = sampled_K.T
    print('sampled_K:', sampled_K.shape)

    rescaling_coefficients = np.sqrt(np.ones(num_samples) / (num_samples * sampled_row_probability))
    print('rescaling_coefficients:', rescaling_coefficients.shape)

    ######## Start Richardson iterations. ##########

    # Compute constant that is used in every step.
    SK = np.einsum('i,ij->ij', rescaling_coefficients, sampled_K)
    print('SK:', SK.shape)

    Sb = rescaling_coefficients * np.ones(SK.shape[0])
    print('Sb:', Sb.shape)
    del sampled_K, sampled_row_probability, rescaling_coefficients

    KtStSb = SK.T @ Sb
    KtStSb = np.reshape(KtStSb, (len(KtStSb), 1))
    print('KtStSb:', KtStSb.shape)
    del Sb

    # Compute decomposition of M^+
    factors_Sigma = []
    factors_Vt = []
    for factor in factors:
        u, s, vt = np.linalg.svd(factor, full_matrices=True)
        factors_Sigma.append(s)
        factors_Vt.append(vt)
    # Constuct diagonal matrix (as a vector)
    D = np.ones(1)
    for Sigma in factors_Sigma:
        D = np.kron(D, Sigma * Sigma)
    D += l2_regularization * np.ones(D.shape[0])
    D = 1.0 / D
    D = np.reshape(D, (len(D), 1))
    del factors_Sigma

    # Initialize iterate
    x = np.zeros(num_columns)
    x = np.reshape(x, (len(x), 1))

    for t in range(100):
        y = SK.T @ (SK @ x) + (l2_regularization * x) - KtStSb
        tmp = kron_mat_mult(factors_Vt, y)
        tmp = D * tmp
        tmp = kron_mat_mult([Vt.T for Vt in factors_Vt], tmp)
        z = x - (1 - epsilon**0.5) * tmp
        if t > 0:
            rre = np.linalg.norm(z - x) / np.linalg.norm(x)
        else:
            rre = 100000
        x = z
        #print(' - step', t, 'rre', rre)
        if rre < 1e-9:
            break
    #print('x:', x)
    print('time:', time.process_time() - start)
    print('loss:', loss_function(factors, l2_regularization, x))

parser = argparse.ArgumentParser()
parser.add_argument('--ndim', type=int)
parser.add_argument('--rows', type=int)
parser.add_argument('--cols', type=int)
parser.add_argument('--alg', type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--l2_regularization', type=float, default=1e-3)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=1.0)

def main():
    print('###################################')
    print(datetime.datetime.now())

    args = parser.parse_args()
    print(args)

    seed = args.seed
    ndim = args.ndim
    num_rows = args.rows
    num_cols = args.cols
    algorithm = args.alg

    l2_regularization = args.l2_regularization

    epsilon = args.epsilon
    delta = args.delta
    alpha = args.alpha

    factors = create_regression_experiment_01(ndim, num_rows, num_cols, seed)

    if algorithm == 1:
        kronecker_regression_algorithm_01(factors, l2_regularization)
    elif algorithm == 2:
        kronecker_regression_algorithm_02(factors, l2_regularization)
    elif algorithm == 3:
        kronecker_regression_algorithm_03(factors, l2_regularization)
    elif algorithm == 4:
        kronecker_regression_algorithm_04(factors, l2_regularization, epsilon, delta, alpha)
    elif algorithm == 5:
        kronecker_regression_algorithm_05(factors, l2_regularization, epsilon, delta, alpha)
    else:
        print('Invalid algorithm:', algorithm)

main()
