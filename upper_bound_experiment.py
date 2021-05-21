import numpy as np
import matplotlib.pyplot as plt

def get_ridge_leverage_score_matrix(A, l2_regularization=0.0):
    I = np.identity(A.shape[1])
    return A @ np.linalg.pinv(A.T @ A + l2_regularization * I) @ A.T

def get_eigs(A):
    return sorted([round(eig.real + 1e-10, 8) for eig in np.linalg.eig(A)[0]])

# Compute sampling distributions after removing entries and compare using:
# 1) leverage scores before removing rows
# 2) our upper bound 1
# 3) uniform random sampling
# Here, min beta is \beta^* = \min(new_p_i / correct_p_i)
def random_gaussian_design_matrix_min_betas(n, d, l2_regularization, percentage_to_remove, seed):
    np.random.seed(seed)
    A = np.random.rand(n, d)
    for i in range(n):
        for j in range(d):
            A[i,j] = np.random.normal(0, 1)
    # Destort rows to make them very non-uniform.
    for i in range(20):
        A[i,:] *= 10

    ridge_scores = get_ridge_leverage_score_matrix(A, l2_regularization)

    row_indices = [i for i in range(n)]
    np.random.shuffle(row_indices)
    last_remove_index = int(percentage_to_remove * n)
    S, S_bar = [], []
    for i in range(n):
        if i < last_remove_index:
            S_bar.append(row_indices[i])
        else:
            S.append(row_indices[i])

    A_S = A[S,:]
    A_S_bar = A[S_bar,:]
    L_S = get_ridge_leverage_score_matrix(A_S, l2_regularization)
    L_S_bar = get_ridge_leverage_score_matrix(A_S_bar, l2_regularization)

    eigenvalues_L_S_bar = get_eigs(L_S_bar)
    lambda_max = max(eigenvalues_L_S_bar)
    c = pow(1.0 - lambda_max, -1)

    sum_removed_leverage_scores = 0
    for i in range(len(S_bar)):
        sum_removed_leverage_scores += ridge_scores[i, i]
    # *********************
    # Exploring what the best multiplicative update weight is.
    # *********************
    c = sum_removed_leverage_scores
    #print('lambda_max:', lambda_max, 'c:', c)

    old_scores = []
    new_scores = []
    upper_bounds_1 = []
    uniform_scores = []
    old_scores_sum = 0.0
    new_scores_sum = 0.0
    upper_bounds_1_sum = 0.0
    uniform_score_sum = 0.0

    for ni in range(len(S)):
        i = S[ni]
        old_score = ridge_scores[i, i]
        old_scores.append(old_score)
        old_scores_sum += old_score

        new_score = L_S[ni, ni]
        new_scores.append(L_S[ni, ni])
        new_scores_sum += new_score

        tmp_sum = 0.0
        for j in S_bar:
            tmp_sum += ridge_scores[i, j]**2
        # ************************************************
        # Try new formulas for the upper bound estimate here.
        # ************************************************
        upper_bound_1 = min(1, old_score + c * tmp_sum)
        upper_bounds_1.append(upper_bound_1)
        upper_bounds_1_sum += upper_bound_1

        uniform_score = 1.0
        uniform_scores.append(uniform_score)
        uniform_score_sum += uniform_score

    #print(old_scores_sum)
    #print(new_scores_sum)
    #print(upper_bounds_1_sum)
    all_probs = []
    min_beta_using_old = 1.0
    min_beta_using_upper_bound_1 = 1.0
    min_beta_using_uniform = 1.0
    for i in range(len(old_scores)):
        old_prob = old_scores[i] / old_scores_sum
        new_prob = new_scores[i] / new_scores_sum
        upper_bound_1_prob = upper_bounds_1[i] / upper_bounds_1_sum
        uniform_prob = uniform_scores[i] / uniform_score_sum

        min_beta_using_old = min(min_beta_using_old, old_prob / new_prob)
        min_beta_using_upper_bound_1 = min(min_beta_using_upper_bound_1, upper_bound_1_prob / new_prob)
        min_beta_using_uniform = min(min_beta_using_uniform, uniform_prob / new_prob)

    return min_beta_using_old, min_beta_using_upper_bound_1, min_beta_using_uniform

def run_full_random_gaussian_beta_experiment():
    n = 1000
    d = 10
    l2_regularization = 0.001

    num_trials = 5
    step_size = 5

    x_percentage = []
    y_min_beta_using_old_mean = []
    y_min_beta_using_upper_bound_1_mean = []
    y_min_beta_using_uniform_mean = []

    y_min_beta_using_old_std = []
    y_min_beta_using_upper_bound_1_std = []
    y_min_beta_using_uniform_std = []

    for i in range(1, 95 + 1, step_size):
        percentage_to_remove = i / 100.0
        min_betas_using_old = []
        min_betas_using_upper_bound_1 = []
        min_betas_using_uniform = []
        for seed in range(num_trials):
            betas = random_gaussian_design_matrix_min_betas(n, d, l2_regularization, percentage_to_remove, seed)
            print(i, seed, betas)
            min_betas_using_old.append(betas[0]**(-1))
            min_betas_using_upper_bound_1.append(betas[1]**(-1))
            min_betas_using_uniform.append(betas[2]**(-1))

        x_percentage.append(percentage_to_remove)
        y_min_beta_using_old_mean.append(np.mean(min_betas_using_old))
        y_min_beta_using_upper_bound_1_mean.append(np.mean(min_betas_using_upper_bound_1))
        y_min_beta_using_uniform_mean.append(np.mean(min_betas_using_uniform))

        y_min_beta_using_old_std.append(np.std(min_betas_using_old))
        y_min_beta_using_upper_bound_1_std.append(np.std(min_betas_using_upper_bound_1))
        y_min_beta_using_uniform_std.append(np.std(min_betas_using_uniform))

    plt.errorbar(x_percentage, y_min_beta_using_old_mean, yerr=y_min_beta_using_old_std, label='leverage_scores')
    plt.errorbar(x_percentage, y_min_beta_using_upper_bound_1_mean, yerr=y_min_beta_using_upper_bound_1_std, label='upper_bound')
    plt.errorbar(x_percentage, y_min_beta_using_uniform_mean, yerr=y_min_beta_using_uniform_std, label='uniform')
    plt.legend()
    plt.show()

def main():
    run_full_random_gaussian_beta_experiment()

main()
