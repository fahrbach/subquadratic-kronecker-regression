import numpy as np
import tensorly as tl
import os
import time
import matplotlib.pyplot as plt
import dataclasses

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
    data_handler.generate_random_tucker(shape=(800, 800, 800),
            rank=(8, 8, 8), random_state=1234)

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (8, 8, 8)
    config.l2_regularization_strength = 0.0

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS-Richardson-numpy'
    #config.algorithm = 'ALS-DJSSW19-numpy'

    config.epsilon = 0.1
    config.downsampling_ratio = 1.0

    config.max_num_steps = 10
    config.rre_gap_tol = 0.0
    config.verbose = False
    print(config)

    # Initialize output log file.
    filename_prefix = 'synthetic'
    output_file = init_output_file(data_handler, config, filename_prefix)

    Y = data_handler.tensor
    Y /= np.max(Y)
    noise = np.random.normal(0, 0.1, config.input_shape)
    Y += noise
    #X_tucker = tucker_als(Y, config, output_file)

    # HOOI ----------------
    # Note: Prints n - 2 logs where n=num_steps
    start_time = time.time()
    core, tucker_factors = tucker(Y, rank=config.rank,
            n_iter_max=config.max_num_steps, init='random',
            tol=config.rre_gap_tol, verbose=True)
    end_time = time.time()
    print('HOOI total time:', end_time - start_time)
    print('HOOI avg time:', (end_time - start_time) / config.max_num_steps)

    old_num_steps = config.max_num_steps
    config.max_num_steps = 0
    X_tucker = tucker_als(Y, config)
    X_tucker.factors = tucker_factors
    X_tucker.core = core
    config.max_num_steps = old_num_steps
    loss = ComputeLossTerms(X_tucker, Y, config.l2_regularization_strength,
                    np.linalg.norm(Y), Y.size)
    print('HOOI RRE:', loss.rre)

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
#
# Experiment for ICML 2022 submission:
# - (250, 250, 3) original image
# - (50, 50, 3) core shape
# - run for 5/10/20 steps until convergence
# - cat is a good candidate so far
# - lambdas: (0.1, 0.3, 1, 3, 10, 30, 100)
# 
# ==============================================================================
def run_image_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_image('data/images/nyc.jpg', resize_shape=(500, 500))
    #data_handler.load_image('data/images/nyc.jpg', resize_shape=(2000, 1280))
    #data_handler.load_image('data/images/tucan.jpg', resize_shape=(1000, 1000))
    #data_handler.load_image('data/images/building.jpg', resize_shape=(250, 250))
    #data_handler.load_image('data/images/cat.jpg', resize_shape=(250, 250))

    config = AlgorithmConfig()

    config.input_shape = data_handler.tensor.shape
    #config.rank = (50, 50, 3)
    config.rank = (10, 10, 3)
    #config.rank = (100, 100, 3)
    #config.l2_regularization_strength = 0.001

    config.algorithm = 'ALS'
    #config.algorithm = 'ALS-RS'
    config.algorithm = 'ALS-RS-numpy'
    config.downsampling_ratio = 0.1
    config.l2_regularization_strength = 0.001
    config.verbose = False
    config.max_num_steps = 10
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    X_tucker = tucker_als(Y, config, output_file, X_tucker=None)

    # Plotting the original and reconstruction from the decompositions
    """
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
    """

    #plt.imshow(to_image(Y))
    #plt.axis('off')
    #plt.savefig('output_figures/cat_original.png', bbox_inches='tight', dpi=256)

    plt.imshow(to_image(tl.tucker_to_tensor(X_tucker)))
    plt.axis('off')
    plt.savefig('output_figures/building' + '_steps-' + str(config.max_num_steps) + '_lambda-' + str(config.l2_regularization_strength) + '.png', bbox_inches='tight', dpi=256)
    plt.show()

# Compute effective dimension in current core update problem
def compute_effective_dimension(X_tucker, l2_regularization_strength):
    factors = X_tucker.factors
    S = np.ones(1)
    for factor in factors:
        U, Sigma, VT = np.linalg.svd(factor)
        S = np.kron(S, Sigma)
    #print(S.shape)
    numerators = S * S
    denominators = S * S + l2_regularization_strength * np.ones(S.shape)
    ans = np.sum(numerators / denominators)
    #print(S[:10])
    #print(numerators[:10])
    #print(ans)
    return ans

# TODO:
# - Run with 20 steps
# - Run with (50, 50, 3) core size?
# - Standardize max/min on axes if we're going to plot these side-by-side
def run_effective_dimension_image_experiment():
    data_handler = TensorDataHandler()
    data_handler.load_image('data/images/nyc.jpg', resize_shape=(500, 500))
    #data_handler.load_image('data/images/mina.jpg', resize_shape=(500, 320))
    #data_handler.load_image('data/images/cat.jpg', resize_shape=(500, 500))

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (25, 25, 3)
    print(config)
    config.verbose = False
    config.max_num_steps = 20
    config.rre_gap_tol = 1e-100

    """
    data_handler = TensorDataHandler()
    data_handler.generate_random_tucker(shape=(100, 100, 100),
            rank=(20, 20, 20), random_state=1234)

    config.input_shape = data_handler.tensor.shape
    config.rank = (5, 5, 5)
    config.algorithm = 'ALS'
    config.rre_gap_tol = 1e-100
    config.max_num_steps = 100
    config.verbose = False
    print(config)
    """

    Y = data_handler.tensor

    # Face image
    #Y = tl.tensor(zoom(face(), (0.3, 0.3, 1)), dtype='float64')
    #Y /= 256

    # TODO: Add markers, move color to axis

    trials = 10   # probably gives smoother result
    l2_regularizations = []
    rres_mean = []
    rres_stdev = []
    deffs_mean = []
    deffs_stdev = []
    min_exp = -4
    max_exp = 3
    num_buckets = 50   # 100 was too much, 50 seems good
    for k in range(num_buckets + 1):
        alpha = float(k)/num_buckets
        exp = min_exp + alpha*(max_exp - min_exp)
        config.l2_regularization_strength = pow(10, exp)

        rres = []
        deffs = []
        for t in range(trials):
            config.random_seed = t
            #output_file = init_output_file(data_handler, config)
            output_file = None
            X_tucker = tucker_als(Y, config, output_file, X_tucker=None)

            loss = ComputeLossTerms(X_tucker, Y, config.l2_regularization_strength,
                    np.linalg.norm(Y), Y.size)
            rres.append(loss.rre)
            deffs.append(compute_effective_dimension(X_tucker, config.l2_regularization_strength))

        l2_regularizations.append(config.l2_regularization_strength)
        rres_mean.append(np.mean(rres))
        rres_stdev.append(np.std(rres))
        deffs_mean.append(np.mean(deffs))
        deffs_stdev.append(np.std(deffs))

    print('steps:', config.max_num_steps)
    print(l2_regularizations)
    print(rres_mean)
    print(rres_stdev)
    print(deffs_mean)
    print(deffs_stdev)

    fig, ax1 = plt.subplots()

    # loss plot
    color = 'tab:blue'
    ax1.set_xlabel('L2 regularization strength (λ)')
    ax1.set_ylabel('Relative reconstruction error (RRE)')
    ax1.set_ylim([0, 1])
    l1 = ax1.errorbar(l2_regularizations, rres_mean, rres_stdev, color=color, label='RRE')

    ax2 = ax1.twinx()

    color = 'tab:orange'

    ax2.set_ylabel('Effective dimension')
    ax2.set_ylim([0, np.prod(config.rank)])
    l2 = ax2.errorbar(l2_regularizations, deffs_mean, deffs_stdev, color=color, label='Effective dimension')

    #plt.title('RRE and DEFF as function of L2 regularization strength')

    plt.xscale('log')
    plt.legend([l1, l2], ['RRE', 'Effective dimension'])
    
    plt.savefig('output_figures/rre_deff' + '_steps-' + str(config.max_num_steps) + '_trials-' + str(trials) + '_num_buckets-' + str(num_buckets) + '.png', bbox_inches='tight', dpi=256)

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

def run_new_video_experiment():
    # TODO(fahrbach): Add resize options. Use first 100 frames for now.
    data_handler = TensorDataHandler()
    data_handler.load_video('data/videos/walking_past_camera.mp4')
    # Resizes to (100, 100, 100, 3)
    #data_handler.tensor = data_handler.tensor[0:100, 0:100, 0:100, :]
    print('before:', data_handler.tensor.shape)
    data_handler.tensor = zoom(data_handler.tensor, [1, 10, 10, 1])
    data_handler.tensor = data_handler.tensor[0:100, :, :, :]
    print('after:', data_handler.tensor.shape)

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (5, 5, 5, 2)
    config.algorithm = 'ALS'
    config.max_num_steps = 1
    #config.algorithm = 'ALS-RS'
    print(config)

    output_file = init_output_file(data_handler, config)

    Y = data_handler.tensor
    print(Y.shape)
    X_tucker = tucker_als(Y, config, output_file)

def run_kron_mat_mul_vs_hooi():
    # Run our algorithm
    data_handler = TensorDataHandler()
    data_handler.generate_random_tucker(shape=(500, 500, 500),
            rank=(20, 20, 20), random_state=1234)
    # TODO(fahrbach): Add Gaussian noise?

    config = AlgorithmConfig()
    config.input_shape = data_handler.tensor.shape
    config.rank = (10, 10, 10)
    config.algorithm = 'ALS'
    config.max_num_steps = 2
    config.rre_gap_tol = 0
    config.l2_regularization_strength = 0.0
    config.verbose = True
    print(config)

    Y = data_handler.tensor
    Y[(0, 0, 0)] = 0

    start_time = time.time()
    X_tucker = tucker_als(Y, config)
    end_time = time.time()
    print('ALS total time:', end_time - start_time)
    print('ALS avg time:', (end_time - start_time) / config.max_num_steps)
    loss = ComputeLossTerms(X_tucker, Y, config.l2_regularization_strength,
                    np.linalg.norm(Y), Y.size)
    print('ALS RRE:', loss.rre)
    print()
    print('######################')
    print()

    data_handler = TensorDataHandler()

    data_handler.generate_random_tucker(shape=(500, 500, 500),
            rank=(20, 20, 20), random_state=1234)
    Y = data_handler.tensor
    Y[(0, 0, 0)] = 0

    # HOOI ----------------
    # Note: Prints n - 2 logs where n=num_steps
    start_time = time.time()
    core, tucker_factors = tucker(Y, rank=config.rank,
            n_iter_max=config.max_num_steps, init='random', tol=None, verbose=True)
    end_time = time.time()
    print('HOOI total time:', end_time - start_time)
    print('HOOI avg time:', (end_time - start_time) / config.max_num_steps)

    old_num_steps = config.max_num_steps
    config.max_num_steps = 0
    X_tucker = tucker_als(Y, config)
    X_tucker.factors = tucker_factors
    X_tucker.core = core
    config.max_num_steps = old_num_steps
    loss = ComputeLossTerms(X_tucker, Y, config.l2_regularization_strength,
                    np.linalg.norm(Y), Y.size)
    print('HOOI RRE:', loss.rre)

# Originally size (2493, 1080, 1920, 3)
def preprocess_video():
    """
    data_handler = TensorDataHandler()
    data_handler.load_video('data/videos/walking_past_camera.mp4')
    data_handler.tensor = data_handler.tensor[:, 0:100, 0:100, :]
    num_elements = data_handler.tensor.size
    print(num_elements)
    np.savetxt('walking_past_camera_tensor_x_100_100_x.gz', tl.tensor_to_vec(data_handler.tensor), fmt='%i')
    """
    data_handler = TensorDataHandler()
    data_handler.load_video('data/videos/walking_past_camera.mp4')
    data_handler.tensor = data_handler.tensor[:, 0:100, 0:100, :]
    print('trimmed')
    num_elements = data_handler.tensor.size
    print('num_elements:', num_elements)
    v = np.reshape(data_handler.tensor, (num_elements))
    print(v.shape)
    print(type(v))
    np.save('walking_past_camera_tensor_x_100_100_x.npy', v)
    #data_handler.tensor.save('walking_past_camera_tensor_x_100_100_x.npy', dtype=int)

def run_sampling_methods_experiment():
    #preprocess_video()
    video_tensor = np.load('walking_past_camera_tensor_x_100_100_x.npy')
    print(video_tensor.shape)
    print('num_elements:', video_tensor.size)  # 74790000
    video_tensor = np.reshape(video_tensor, (2493, 100, 100, 3))
    print(video_tensor.shape)

    for n in range(1, 10 + 1):
        n_frames = n*100
        Y = video_tensor[0:n_frames, :, :, :]
        print('######## starting trial:', n)
        print('shape:', Y.shape)
        config = AlgorithmConfig()
        config.input_shape = Y.shape
        config.l2_regularization_strength = 0.0
        config.rank = (1 + n, 4, 4, 3)
        config.downsampling_ratio = 0.01
        config.verbose = False
        config.max_num_steps = 5
        config.rre_gap_tol = 1e-9
        #config.algorithm = 'ALS'
        config.algorithm = 'ALS-RS-Richardson'
        #config.algorithm = 'ALS-DJSSW19'
        print(config)

        output_file = init_output_file(None, config, 'output/videos/walking')
        X_tucker = tucker_als(Y, config, output_file)

# New synthetic
def new_synthetic():
    for n, r in [(500, 4)]:
        print('Trial:', n, r)
        data_handler = TensorDataHandler()
        data_handler.generate_random_tucker(shape=(n, n, n),
                rank=(50, 50, 50), random_state=1234)

        config = AlgorithmConfig()
        config.input_shape = data_handler.tensor.shape
        config.rank = (r, r, r)
        config.downsampling_ratio = 1 #0.01
        config.verbose = False
        config.max_num_steps = 5
        config.rre_gap_tol = 1e-100
        config.l2_regularization_strength = 0.0

        #config.algorithm = 'ALS'
        #config.algorithm = 'ALS-RS'
        config.algorithm = 'ALS-RS-numpy'
        #config.algorithm = 'ALS-RS-Richardson'
        #config.algorithm = 'ALS-DJSSW19'
        print(config)

        output_file = init_output_file(data_handler, config, 'cubic')

        Y = data_handler.tensor
        Y -= np.mean(Y)
        Y /= np.std(Y)
        noise = np.random.normal(0, 0.01, config.input_shape)
        Y += noise
        X_tucker = tucker_als(Y, config, output_file)

def main():
    #run_synthetic_experiment()
    # run_synthetic_shapes_experiment()
    # run_cardiac_mri_experiment()
    #run_image_experiment()
    # run_video_experiment()

    #run_effective_dimension_image_experiment()
    #run_kron_mat_mul_vs_hooi()
    #run_new_video_experiment()

    # VERY GOOD
    #run_sampling_methods_experiment()

    #new_synthetic()

    # NeurIPS 2022
    run_synthetic_experiment()

main()
