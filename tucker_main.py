import numpy as np
import tensorly as tl
from tensorly.random import random_tucker

def tensor_to_vector(tensor):
    num_elements = 1
    for n in tensor.shape:
        num_elements *= n
    return tl.reshape(tensor, (num_elements, 1))

def solve_least_squares(A, b, l2_regularization):
    AtA_lambda = A.T @ A + l2_regularization * np.identity(A.shape[1])
    Atb = A.T @ b
    return np.linalg.solve(AtA_lambda, Atb)

def update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization):
    # See the matricized version of Equation 4.2 in "Tensor Decompositions and
    # Applications."
    Y_matrix = tl.unfold(Y_tensor, factor_index)
    core_matrix = tl.unfold(X_tucker.core, factor_index)
    K_matrix = np.identity(1)
    for n in range(len(X_tucker.core.shape)):
        if n == factor_index: continue
        # Important: Note the ordering of this Kronecker product. For some
        # reason, it disagrees with Equation 4.2 in the reference above... Do
        # they have a typo?
        K_matrix = np.kron(K_matrix, X_tucker.factors[n])

    # Each row of the current factor matrix is its own least squares problem.
    design_matrix = K_matrix @ core_matrix.T
    for row_index in range(X_tucker.factors[factor_index].shape[0]):
        response_vector = Y_matrix[row_index,:]
        X_tucker.factors[factor_index][row_index,:] = solve_least_squares( \
                design_matrix, response_vector, l2_regularization)

# TODO(fahrbach): Need to avoid O(n * d^2) space bottleneck. Shouldn't compute
# the design matrix explicitly.
def construct_linear_system_for_core_tensor(X_tucker, Y_tensor, l2_regularization):
    d = 1
    for dimension in X_tucker.core.shape:
        d *= dimension
    A = l2_regularization * np.identity(d)

    design_matrix = np.identity(1)
    for n in range(len(X_tucker.core.shape)):
        design_matrix = np.kron(design_matrix, X_tucker.factors[n])
    Y_vector = tensor_to_vector(Y_tensor)
    return design_matrix, Y_vector

def compute_loss(Y_tensor, X_tucker, l2_regularization):
    loss = 0.0
    residual_vector = tensor_to_vector(Y_tensor - tl.tucker_to_tensor(X_tucker))
    loss += np.linalg.norm(residual_vector)**2
    loss += l2_regularization * np.linalg.norm(tensor_to_vector(X_tucker.core))**2
    for n in range(len(X_tucker.factors)):
        loss += l2_regularization * np.linalg.norm(X_tucker.factors[n])**2
    return loss

def main():
    shape = (300, 40, 4)
    rank = (10, 5, 2)
    l2_regularization = 0.00001
    order = len(shape)

    # These settings converge nicely.
    """
    shape = (30, 400, 40, 5)
    rank = (4, 4, 2, 2)
    l2_regularization = 0.0001
    order = len(shape)
    # set Y_tensor[0,0,0,0] = 1
    """

    # --------------------------------------------------------------------------
    # Intialize tensors.
    Y_tucker = random_tucker(shape, rank, random_state=0)
    Y_tensor = tl.tucker_to_tensor(Y_tucker)
    Y_tensor[0,0,0] = 1
    print('Target tensor Y:')
    print(Y_tucker)
    print()

    X_tucker = random_tucker(shape, rank, random_state=2)
    print('Learned tensor X:')
    print(X_tucker)
    print()

    loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
    print('Loss:', loss)
    print()

    for step in range(10000):
        print('Step:', step, '###################################')
        # --------------------------------------------------------------------------
        # Factor matrix updates:
        for factor_index in range(len(X_tucker.core.shape)):
            #print('Updating factor matrix:', factor_index)
            update_factor_matrix(X_tucker, Y_tensor, factor_index, l2_regularization)
            new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
            print('Loss:', new_loss)
            if new_loss > loss:
                print('Loss function increased!!!!!!!!!')
            loss = new_loss
        
        # --------------------------------------------------------------------------
        # Core tensor update:
        #print('Updating core tensor')
        # Constructing the design matrix is killing the algorithm...
        design_matrix, Y_vector = construct_linear_system_for_core_tensor( \
                X_tucker, Y_tensor, l2_regularization)
        y = solve_least_squares(design_matrix, Y_vector, l2_regularization)
        del design_matrix
        # Update core tensor.
        X_tucker.core = tl.reshape(y, rank)
        new_loss = compute_loss(Y_tensor, X_tucker, l2_regularization)
        print('Loss:', new_loss)
        if new_loss > loss:
            print('Loss function increased!!!!!!!!!')
        loss = new_loss

main()
