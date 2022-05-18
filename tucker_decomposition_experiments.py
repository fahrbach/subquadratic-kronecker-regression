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
        print('Loading MRI tensor...')
        data_handler.load_cardiac_mri_data()
        print('Finished.')
        config.input_shape = data_handler.tensor.shape
    elif args.data == 'coil':
        print('Loading COIL-100 tensor...')
        data_handler.load_coil_100()
        print('Finished.')
        config.input_shape = data_handler.tensor.shape
    elif args.data == 'hyperspectral':
        print('Loading hyperspectral tensor...')
        data_handler.load_hyperspectral()
        print('Finished.')
        config.input_shape = data_handler.tensor.shape
    elif args.data == 'movie':
        print('Loading movie tensor...')
        data_handler.load_video('data/videos/walking_past_camera.mp4')
        print('Finished.')
        config.input_shape = data_handler.tensor.shape
    else:
        print('Invalid data:', data)

    print(config)
    X_tucker = tucker_als(data_handler.tensor, config)

main()
