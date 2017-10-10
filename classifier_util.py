__author__ = 'raphey'

import numpy as np
import glob
import cv2


def import_data(directory, full_normalization=False, extra_dim=False):
    """
    Returns a numpy array with all the .png files in a directory.
    arr[i, j, k] is the ith image, vertical coordinate j (careful!), horizontal coordinate k.
    The values are single grayscale integers < 256.
    """

    arr = []
    for image_path in glob.glob(directory + '/*.png'):
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        # gray_img /= 256.0    # Simple scaling doesn't seem to work as well

        if full_normalization:
            # Scale each image to range 0.0 - 1.0
            min_val = gray_img.min()
            max_val = gray_img.max()
            gray_img -= min_val
            gray_img /= (max_val - min_val)
        else:
            gray_img /= 256.0    # Simple scaling

        if extra_dim:                           # Add extra dimension for convolution
            gray_img = np.atleast_3d(gray_img)

        arr.append(gray_img)

    return np.array(arr)


def shuffle_and_split_data(arr, valid_portion, test_portion):
    """
    Shuffles the data and returns three arrays. First is training, second is validation,
    third is testing. Portion arguments are relative to entire size of data.
    """
    np.random.shuffle(arr)
    l = arr.shape[0]
    test_cutoff = l - int(l * test_portion)
    validation_cutoff = test_cutoff - int(l * valid_portion)
    return arr[:validation_cutoff], arr[validation_cutoff: test_cutoff], arr[test_cutoff:]


def combine_and_shuffle_data(cell_training_data, non_cell_training_data):
    """
    Returns a shuffled combination of data and labels for cells and non-cells.
    """
    all_training_labels = np.vstack((np.ones((cell_training_data.shape[0], 1)),
                                     np.zeros((non_cell_training_data.shape[0], 1))))
    all_training_data = np.vstack((cell_training_data, non_cell_training_data))
    p = np.random.permutation(len(all_training_labels))
    return all_training_data[p], all_training_labels[p]     # Using np's cool indexing


def flatten_images(img_data):
    """
    Reshape series of images down a dimension to use with a simple regression or neural network
    """
    _, h, w = img_data.shape
    return img_data.reshape(-1, h * w)


def prepare_data(c_data, nc_data, valid_portion=0.1, test_portion=0.1, flat=False):
    """
    Returns three items, corresponding to training data, validation data, and testing data.
    Training data is in the form of (images, labels). Validation and testing data are in the
    form of (cell_images, cell_labels, non_cell_images, non_cell_labels), since we'll want
    separate counts for positive and negative accuracy.
    With flatten on, all image data is converted from shape [28, 28] to [784].
    """
    if flat:
        c_data = flatten_images(c_data)
        nc_data = flatten_images(nc_data)

    # Separately split positive and negative data
    c_train, c_valid, c_test = shuffle_and_split_data(c_data, valid_portion, test_portion)
    nc_train, nc_valid, nc_test = shuffle_and_split_data(nc_data, valid_portion, test_portion)

    train_data_w_labels = combine_and_shuffle_data(c_train, nc_train)

    # Get validation labels and make validation 4-tuple
    c_valid_labels = np.ones(shape=[len(c_valid), 1])
    nc_valid_labels = np.zeros(shape=[len(nc_valid), 1])
    valid_data_w_labels = (c_valid, c_valid_labels, nc_valid, nc_valid_labels)

    # Get testing labels and make testing 4-tuple
    c_test_labels = np.ones(shape=[len(c_test), 1])
    nc_test_labels = np.zeros(shape=[len(nc_test), 1])
    test_data_w_labels = (c_test, c_test_labels, nc_test, nc_test_labels)

    return train_data_w_labels, valid_data_w_labels, test_data_w_labels


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_cost(y_hat, y_actual):
    return 0.5 * sum((y_actual[i] - y_hat[i]) ** 2 for i in range(0, len(y_actual)))[0]


def initialize_weight_array(rows, cols, stddev=None, sigma_cutoff=2.0):

    # Default initialization stddev proportional to input size
    if stddev is None:
        stddev = (1.0 / (rows * cols)) ** 0.5

    weights = []
    while len(weights) < rows * cols:
        new_rand_val = np.random.randn() * stddev
        if abs(new_rand_val) < sigma_cutoff * stddev:
            weights.append(new_rand_val)
    return np.array(weights).reshape(rows, cols)


def accuracy(y_pred, y_actual):
    return np.sum(y_pred == y_actual) / len(y_actual)


def unit_tests():
    cell_data = import_data('training_data/cells')
    non_cell_data = import_data('training_data/non_cells')
    training, validation, testing = prepare_data(cell_data, non_cell_data)

    # train/valid/test tuples are correct length
    assert(len(training) == 2)
    assert(len(validation) == 4)
    assert(len(testing) == 4)

    # Data and labels have matching length
    assert(training[0].shape[0] == training[1].shape[0])
    assert(validation[0].shape[0] == validation[1].shape[0])
    assert(validation[2].shape[0] == validation[3].shape[0])
    assert(testing[0].shape[0] == testing[1].shape[0])
    assert(testing[2].shape[0] == testing[3].shape[0])

    # Labels are all shape (x, 1)
    assert(training[1].shape[1] == 1)
    assert(validation[1].shape[1] == 1)
    assert(validation[3].shape[1] == 1)
    assert(testing[1].shape[1] == 1)
    assert(testing[3].shape[1] == 1)

    # Positive labels are zero, negative labels are one
    assert(all(x[0] == 1.0 for x in validation[1]))
    assert(all(x[0] == 0.0 for x in validation[3]))
    assert(all(x[0] == 1.0 for x in testing[1]))
    assert(all(x[0] == 0.0 for x in testing[3]))

    # Training labels are a mixture of zeros and ones
    assert(any(x[0] == 1.0 for x in training[1]))
    assert(any(x[0] == 0.0 for x in training[1]))

    # Image pixel data is scaled from 0.0 to 1.0
    assert(all(x.min() == 0.0 and x.max() == 1.0 for x in training[0]))
    assert(all(x.min() == 0.0 and x.max() == 1.0 for x in validation[0]))
    assert(all(x.min() == 0.0 and x.max() == 1.0 for x in validation[2]))
    assert(all(x.min() == 0.0 and x.max() == 1.0 for x in testing[0]))
    assert(all(x.min() == 0.0 and x.max() == 1.0 for x in testing[2]))

    print('Tests pass')

