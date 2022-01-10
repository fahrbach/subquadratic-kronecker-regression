import numpy as np
import tensorly as tl
import os
import time
import matplotlib.pyplot as plt
import dataclasses

from tensor_data_handler import TensorDataHandler
from tucker_als import *

# Initializes and returns output file for logging.
def init_output_file(data_handler, config):
    output_filename = data_handler.output_filename_prefix
    output_filename += '_' + ','.join([str(x) for x in config.input_shape])
    output_filename += '_' + ','.join([str(x) for x in config.rank])
    output_filename += '_' + "{:.1e}".format(config.l2_regularization_strength)
    output_filename += '_' + config.algorithm
    output_filename += '.txt'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'a')
    
    output_file.write('##############################################\n')
    if data_handler.input_filename != None:
        output_file.write('input_filename: ' + data_handler.input_filename+'\n')
        print('input_filename:', data_handler.input_filename)

    config_dict = dataclasses.asdict(config)
    for key in config_dict:
        output_file.write(str(key) + ': ' + str(config_dict[key]) + '\n')

    # Compute size and compression stats.
    input_size = np.prod(config.input_shape)
    tucker_size = np.prod(config.rank)
    assert(len(config.input_shape) == len(config.rank))
    for n in range(len(config.input_shape)):
        tucker_size += config.input_shape[n] * config.rank[n]
    output_file.write('input_size: ' + str(input_size) + '\n')
    output_file.write('tucker_size: ' + str(tucker_size) + '\n')
    output_file.write('compression: ' + str(tucker_size / input_size) + '\n')
    print('compression:', str(tucker_size / input_size))
    output_file.flush()
    return output_file

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# ==============================================================================
# Synthetic Experiment:
# - Simple tensor decomposition experiment where a tensor Y is randomly generated
#   by a random Tucker decomposition, with one entry set to Y[0,0,0] = 1, so
#   that it can't be fit perfectly.
# - Then we generate a new random Tucker decomposition X (using a different
#   seed), and we try to learn Y.
# - Note: We start to see nice gains from ALG-RS when the tensor has shape
#   ~(1028, 1028, 512) and the rank is (4, 4, 4).
# ==============================================================================
def run_synthetic_experiment():
    data_handler = TensorDataHandler()
    data_handler.generate_random_tucker(shape=(100, 200, 300),
            rank=(10, 20, 30), random_state=1234)

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (2, 4, 6)
    #config.algorithm = 'ALS'
    config.algorithm = 'ALS-RS'
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    Y[(0, 0, 0)] = 0
    X_tucker = tucker_als(Y, config, output_file)

# ==============================================================================
# Synthetic Shapes Experiment:
# - Use Tensorly's built-in shape images.
# - Note: This data can easily scale up, and starts to show the benefit of
#   row sampling. For example, create a shape of dimensions [1024, 1024, 3]
#   and rank [4, 4, 3]. Observe that it's only sampling about 0.1% of the rows.
# ==============================================================================
def run_synthetic_shapes_experiment():
    data_handler = TensorDataHandler()
    pattern = 'circle'  # ['rectangle', 'swiss', 'circle']
    data_handler.generate_synthetic_shape(pattern, 200, 200, n_channels=3)

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (40, 40, 2)
    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS'
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    X_tucker = tucker_als(Y, config, output_file, X_tucker=None)

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
    if data_handler.input_filename != None:
        print('input_filename: ', data_handler.input_filename)
        output_file.write('input_filename: ' + data_handler.input_filename+'\n')

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

# ==============================================================================
# Image Experiments
# - Read 3-way image tensor (x, y, RGB channel).
# ==============================================================================
def run_image_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_image('data/images/nyc.jpg', resize_shape=(500, 320))
    #data_handler.load_image('data/images/nyc.jpg', resize_shape=(2000, 1280))

    config = AlgorithmConfig()

    config.input_shape = data_handler.tensor.shape
    config.rank = (25, 25, 2)
    #config.rank = (100, 100, 3)
    #config.l2_regularization_strength = 0.001

    #config.algorithm = 'ALS'
    config.algorithm = 'ALS-RS'
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    X_tucker = tucker_als(Y, config, output_file, X_tucker=None)

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
# Video Experiments:
# - Read 4-way video tensor (frame, x, y, RGB) of shape (2493, 1080, 1920, 3).
# ==============================================================================
def run_video_experiment():
    # TODO(fahrbach): Add resize options. Use first 100 frames for now.
    data_handler = TensorDataHandler()
    data_handler.load_video('data/videos/walking_past_camera.mp4')
    data_handler.tensor = data_handler.tensor[0:100, 0:100, 0:100, :]

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (5, 5, 5, 2)
    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS'
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    X_tucker = tucker_als(Y, config, output_file)

def main():
    # run_synthetic_experiment()
    # run_synthetic_shapes_experiment()
    # run_cardiac_mri_experiment()
    run_image_experiment()
    # run_video_experiment()


main()
