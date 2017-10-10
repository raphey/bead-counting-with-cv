__author__ = 'raphey'

import numpy as np
import cv2
import tensorflow as tf
from scipy import misc
import os
from cell_classifier_tf_cnn import conv_net
from cell_counter_tf_cnn import *


def find_cells(image_4x):
    """

    """

    # Create variables for CNN model import
    weights, biases, x_input, y_actual, y_pred = tf_model_setup()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore variables from disk.
        saver.restore(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")
        print("TensorFlow model restored.")

        img_tensor = []
        for x in range(19, 1792 - 19):
            for y in range(19, 1792 - 19):
                big_cell_img = image_4x[y - 18: y + 18, x - 18: x + 18]
                cell_img = misc.imresize(big_cell_img, (9, 9))
                scaled_img = cell_img.astype(float) / 256.0
                reshaped_img = scaled_img.reshape(9, 9, 1)
                img_tensor.append(reshaped_img)

        print("Image tensor formed.")

        tf_model_output = sess.run(y_pred, feed_dict={x_input: np.array(img_tensor)}).reshape(1754, 1754, 1)

        print("Probabilities computed.")

        c_probs_lookup = np.zeros(shape=(1792, 1792))

        c_probs = []

        for x in range(1754):
            for y in range(1754):
                p = tf_model_output[x, y, 0]
                c_probs.append((p, x + 19, y + 19))
                c_probs_lookup[y + 19, x + 19] = p

        c_probs.sort()
        c_probs.reverse()

        # Grab valid cells
        cs = []
        for p, x, y in c_probs:
            if p < 0.99:
                break
            if any(p < c_probs_lookup[y + dy, x + dx] for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            if any(distance(c[0], c[1], x, y) < 10 for c in cs):
                continue
            cs.append((x, y))

        return cs


if __name__ == '__main__':

    # Suppress sub-optimal speed warnings from TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Random seed for consistent color display on multiple runs
    np.random.seed(0)

    _, grayscale_image_4x, original_image_4x = read_image('images/test_array_3_hi_res.png')

    droplet_data = find_and_process_droplets(grayscale_image_4x)

    cells = find_cells(grayscale_image_4x)

    cells_by_cluster = group_cells_by_cluster(cells, droplet_data)

    write_output(original_image_4x, droplet_data, cells_by_cluster, save=True)