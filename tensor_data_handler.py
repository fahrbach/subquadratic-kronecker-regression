import numpy as np
from PIL import Image
import tensorly as tl
import scipy.io as sio

def get_output_filename_prefix(input_filename):
    tokens = input_filename.split('/')
    assert(tokens[0] == 'data')
    tokens[0] = 'output'
    output_filename_prefix = '/'.join(tokens)
    return output_filename_prefix.split('.')[0]

class TensorDataHandler:
    def __init__(self):
        self.tensor = None
        self.input_filename = None
        self.output_filename_prefix = None

    def load_random_tucker(self, shape, rank, random_state):
        self.tensor = tl.random.random_tucker(shape, rank, full=True,
                random_state=random_state)
        self.output_filename_prefix = 'output/random_tucker/'

    def load_synthetic_shape(self, pattern='swiss', image_height=20,
            image_width=20, n_channels=None):
         self.tensor = tl.datasets.synthetic.gen_image(pattern, image_height,
                 image_width, n_channels)
         self.output_filename_prefix = 'output/synthetic_shapes/' + pattern

    def load_image(self, input_filename, resize_shape=None):
        self.input_filename = input_filename
        image = Image.open(input_filename)
        if resize_shape:
            image = image.resize((resize_shape[0], resize_shape[1]), Image.ANTIALIAS)
        self.tensor = np.array(image) / 256
        self.output_filename_prefix = get_output_filename_prefix(input_filename)

    def load_cardiac_mri_data(self):
        self.input_filename = 'data/cardiac_mri_data/sol_yxzt_pat1.mat'
        self.tensor = sio.loadmat(self.input_filename)['sol_yxzt']
        self.output_filename_prefix = get_output_filename_prefix(self.input_filename)
