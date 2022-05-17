import argparse
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

# ==============================================================================
# COIL-100
# ==============================================================================
def run_coil_100_experiment():
    print('Loading COIL-100 tensor...')
    data_handler = TensorDataHandler()
    data_handler.load_coil_100()
    print('tensor.shape:', data_handler.tensor.shape)

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (1, 1, 1, 1)

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS-Richardson'
    #config.algorithm = 'ALS-DJSSW19'
    #config.algorithm = 'HOOI'
    #config.algorithm = algorithm

    config.epsilon = 0.1
    config.downsampling_ratio = 1e-3
    config.max_num_samples = 1024

    config.max_num_steps = 10
    config.rre_gap_tol = 0.0
    config.verbose = True
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

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
# Valid: HOOI, ALS, ALS-RS, ALS-DJSSW19
parser.add_argument('--algorithm', type=str)
# Comma separated int list
parser.add_argument('--rank', type=str)
parser.add_argument('--seed', type=int, default=0)
# Downsampling ratio
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--max_num_samples', type=int, default=0)

parser.add_argument('--max_num_steps', type=int, default=5)
parser.add_argument('--rre_gap_tol', type=float, default=0.0)
# Verbose logging
parser.add_argument('--verbose', type=bool, default=False)

def main():
    print('###################################')
    print(datetime.datetime.now())

    args = parser.parse_args()
    print(args)

    config = AlgorithmConfig()
    config.algorithm = args.algorithm
    config.rank = tuple([int(x) for x in args.rank.split(',')])
    config.seed = args.seed
    
    config.epsilon = 0.1
    config.delta = 0.01
    config.downsampling_ratio = args.alpha
    config.max_num_samples = args.max_num_samples

    config.max_num_steps = args.max_num_steps
    config.rre_gap_tol = args.rre_gap_tol
    config.verbose = args.verbose

    data_handler = TensorDataHandler()
    if args.data == 'mri':
        data_handler.load_cardiac_mri_data()
        config.input_shape = data_handler.tensor.shape
    elif args.data == 'coil':
        print('Loading COIL-100 tensor...')
        data_handler.load_coil_100()
        print('Finished.')
        config.input_shape = data_handler.tensor.shape
    else:
        print('Invalid data:', data)

    print(config)
    X_tucker = tucker_als(data_handler.tensor, config)

main()
