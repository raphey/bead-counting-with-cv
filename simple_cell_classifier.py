__author__ = 'raphey'

import numpy as np
import glob
import cv2


def import_data(directory):
    """
    Returns a numpy array with all the .png files in a directory.
    arr[i, j, k] is the ith image, vertical coordinate j (careful!), horizontal coordinate k.
    The values are single grayscale integers < 256.
    """
    TODO: scale individual images to 0.0-1.0
    arr = []
    for image_path in glob.glob(directory + '/*.png'):
        img = cv2.imread(image_path)
        arr.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return np.array(arr, dtype='float32')


def shuffle_and_split_data(arr, validation_portion=0.1, test_portion=0.1):
    """
    Shuffles the data and returns three arrays. First is training, second is validation,
    third is testing. Portion arguments are relative to entire size of data.
    """
    np.random.shuffle(arr)
    l = arr.shape[0]
    test_cutoff = l - int(l * test_portion)
    validation_cutoff = test_cutoff - int(l * validation_portion)
    return arr[:validation_cutoff], arr[validation_cutoff: test_cutoff], arr[test_cutoff:]


def label_and_combine_data(cell_training_data, non_cell_training_data):
    """
    Returns a shuffled combination of data and labels for cells and non-cells.
    """
    all_training_labels = np.vstack((np.ones((cell_training_data.shape[0], 1)),
                                     np.zeros((non_cell_training_data.shape[0], 1))))
    all_training_data = np.vstack((cell_training_data, non_cell_training_data))
    p = np.random.permutation(len(all_training_labels))
    return all_training_data[p], all_training_labels[p]     # Using np's cool indexing


def flatten_image_data(img_data):
    """
    Reshape data down a dimension to use with a simple regression or neural network
    """
    _, h, w = img_data.shape
    return img_data.reshape(-1, h * w)


cell_data = import_data('training_data/cells')
cell_training, cell_validation, cell_testing = shuffle_and_split_data(cell_data)

non_cell_data = import_data('training_data/non_cells')
non_cell_training, non_cell_validation, non_cell_testing = shuffle_and_split_data(non_cell_data)

training_data, training_labels = label_and_combine_data(cell_training, non_cell_training)


# Reshape data from 28x28 to 784
flat_training_data = flatten_image_data(training_data)

flat_cell_validation_data = flatten_image_data(cell_validation)
flat_non_cell_validation_data = flatten_image_data(non_cell_validation)

flat_cell_testing_data = flatten_image_data(cell_testing)
flat_non_cell_testing_data = flatten_image_data(non_cell_testing)