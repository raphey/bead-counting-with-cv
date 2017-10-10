__author__ = 'raphey'

import numpy as np
import glob
import cv2
from classifier_util import *


def fwd_pass(x, w, b):
    return sigmoid(np.dot(x, w) + b)


def make_prediction(x, w, b):
    return fwd_pass(x, w, b).round()


def train(save=False, verbose=False):

    cell_data = import_data('training_data/set1/cells', full_normalization=True)
    non_cell_data = import_data('training_data/set1/non_cells', full_normalization=True)
    training, validation, testing = prepare_data(cell_data, non_cell_data, flat=True)

    weight = initialize_weight_array(784, 1)
    bias = 0.0
    alpha = 0.01
    epochs = 1200
    batch_size = 12
    num_batches = len(training[0]) // batch_size

    for e in range(epochs):
        cost = 0.0
        for b in range(num_batches):
            start_index = batch_size * b
            batch_x = training[0][start_index:start_index + batch_size]
            batch_y = training[1][start_index:start_index + batch_size]
            logits = fwd_pass(batch_x, weight, bias)
            cost += logit_cost(logits, batch_y)
            y_diff = batch_y - logits
            bias += alpha / batch_size * (y_diff * logits * (1 - logits)).sum(axis=0)
            weight += alpha / batch_size * np.dot(batch_x.T, y_diff * logits * (1 - logits))

        pos_valid_acc = accuracy(make_prediction(validation[0], weight, bias), validation[1])
        neg_valid_acc = accuracy(make_prediction(validation[2], weight, bias), validation[3])
        if verbose and (e + 1) % 50 == 0:
            print('Epoch: {:>4}/{}   Training cost: {:<5.1f}   Pos. & neg. val. acc.: {:.3f},  {:.3f}'.format(
                  e + 1, epochs, cost, pos_valid_acc, neg_valid_acc))
    if save:
        np.save('classifier_data/simple_cell_classifier/weight.npy', weight)
        np.save('classifier_data/simple_cell_classifier/bias.npy', bias)

    pos_test_acc = accuracy(make_prediction(testing[0], weight, bias), testing[1])
    neg_test_acc = accuracy(make_prediction(testing[2], weight, bias), testing[3])

    print('Training complete. Positive & negative testing accuracy: {:.3f},  {:.3f}'.format(
          pos_test_acc, neg_test_acc))
train(save=False, verbose=True)

