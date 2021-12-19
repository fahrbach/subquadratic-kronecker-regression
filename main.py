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
from tucker_als import *

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
    config.l2_regularization_strength = 0.001

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS'
    #config.verbose = False

    config.rre_gap_tol = None
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