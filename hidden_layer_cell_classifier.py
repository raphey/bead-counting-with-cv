__author__ = 'raphey'

import numpy as np
import glob
import cv2
from classifier_util import *


def fwd_pass(x, w1, b1, w2, b2):
    sig_l1 = sigmoid(np.dot(x, w1) + b1)
    y_hat = sigmoid(np.dot(sig_l1, w2) + b2)
    return sig_l1, y_hat


def make_prediction(x, w1, b1, w2, b2):
    _, logits = fwd_pass(x, w1, b1, w2, b2)
    return logits.round()


def train(save=False, verbose=False):

    cell_data = import_data('training_data/cells')
    non_cell_data = import_data('training_data/non_cells')
    training, validation, testing = prepare_data(cell_data, non_cell_data, flat=True)

    weight1 = initialize_weight_array(784, 25)
    bias1 = 0.0
    weight2 = initialize_weight_array(25, 1)
    bias2 = 0.0

    alpha = 0.005
    epochs = 2000
    batch_size = 12
    num_batches = len(training[0]) // batch_size

    for e in range(epochs):
        cost = 0.0
        for b in range(num_batches):
            start_index = batch_size * b
            batch_x = training[0][start_index:start_index + batch_size]
            batch_y = training[1][start_index:start_index + batch_size]

            sig_l1, logits = fwd_pass(batch_x, weight1, bias1, weight2, bias2)

            y_diff = batch_y - logits

            cost += logit_cost(logits, batch_y)

            delta_l1o = np.dot(y_diff, weight2.T) * sig_l1 * (1 - sig_l1)

            delta_l2o = y_diff * logits * (1 - logits)

            weight1 += alpha / batch_size * np.dot(batch_x.T, delta_l1o)
            weight2 += alpha / batch_size * np.dot(sig_l1.T, delta_l2o)
            bias1 += alpha / batch_size * delta_l1o.sum(axis=0)
            bias2 += alpha / batch_size * delta_l2o.sum(axis=0)

        pos_valid_acc = accuracy(make_prediction(validation[0], weight1, bias1, weight2, bias2), validation[1])
        neg_valid_acc = accuracy(make_prediction(validation[2], weight1, bias1, weight2, bias2), validation[3])
        if verbose and (e + 1) % 50 == 0:
            print('Epoch: {:>4}/{}   Training cost: {:<5.1f}   Pos. & neg. val. acc.: {:.3f},  {:.3f}'.format(
                  e + 1, epochs, cost, pos_valid_acc, neg_valid_acc))
    if save:
        np.save('classifier_data/hidden_layer_classifier_weight1.npy', weight1)
        np.save('classifier_data/hidden_layer_classifier_bias1.npy', bias1)
        np.save('classifier_data/hidden_layer_classifier_weight2.npy', weight2)
        np.save('classifier_data/hidden_layer_classifier_bias2.npy', bias2)

    pos_test_acc = accuracy(make_prediction(testing[0], weight1, bias1, weight2, bias2), testing[1])
    neg_test_acc = accuracy(make_prediction(testing[2], weight1, bias1, weight2, bias2), testing[3])

    print('Training complete. Positive & negative testing accuracy: {:.3f},  {:.3f}'.format(
          pos_test_acc, neg_test_acc))

if __name__ == '__main__':
    train(save=True, verbose=True)

