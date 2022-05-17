import numpy as np
import tensorly as tl
import os
import time
import matplotlib.pyplot as plt
import dataclasses
import datetime

from scipy.misc import face
from scipy.ndimage import zoom

from tensor_data_handler import TensorDataHandler
from tucker_als import *

# Initializes and returns output file for logging.
def init_output_file(data_handler, config, prefix=None):
    output_filename = ''
    if data_handler:
        output_filename += data_handler.output_filename_prefix
    if prefix:
        output_filename += prefix
    output_filename += '_' + ','.join([str(x) for x in config.input_shape])
    output_filename += '_' + ','.join([str(x) for x in config.rank])
    output_filename += '_' + "{:.1e}".format(config.l2_regularization_strength)
    output_filename += '_' + config.algorithm
    output_filename += '.txt'

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'a')
    
    output_file.write('##############################################\n')
    output_file.write(str(datetime.datetime.now()) + '\n')
    if data_handler != None and data_handler.input_filename != None:
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

# ==============================================================================
# Cardiac MRI Experiment:
# - Read tensor with shape (256, 256, 14, 20) corresponding to (x, y, z, time).
# ==============================================================================
def run_cardiac_mri_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_cardiac_mri_data()

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (20, 10, 3, 3)
    config.l2_regularization_strength = 0.0

    #config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS-Richardson'
    #config.algorithm = 'ALS-DJSSW19'
    config.algorithm = 'HOOI'

    config.epsilon = 0.1
    config.downsampling_ratio = 0.01
    #config.max_num_samples = 1028

    config.max_num_steps = 3
    config.rre_gap_tol = 0.0
    config.verbose = False
    print(config)

    output_file = init_output_file(data_handler, config)

    X_tucker = tucker_als(data_handler.tensor, config, output_file)

# ==============================================================================
# Video Experiments:
# - Read 4-way video tensor (frame, x, y, RGB) of shape (2493, 1080, 1920, 3).
# ==============================================================================
def run_video_experiment():
    # TODO(fahrbach): Add resize options. Use first 100 frames for now.
    data_handler = TensorDataHandler()
    data_handler.load_video('data/videos/walking_past_camera.mp4')
    print(data_handler.tensor.shape)

    #data_handler.tensor = data_handler.tensor[0:100, 0:100, 0:100, :]

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (20, 10, 3, 3)
    config.l2_regularization_strength = 0.0

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS-Richardson'
    #config.algorithm = 'ALS-DJSSW19'

    config.epsilon = 0.1
    config.downsampling_ratio = 0.01
    #config.max_num_samples = 1028

    config.max_num_steps = 1
    config.rre_gap_tol = 0.0
    config.verbose = False
    print(config)

def main():
    run_cardiac_mri_experiment()
    #run_video_experiment()

main()
