__author__ = 'raphey'

import numpy as np
import cv2
import tensorflow as tf
from scipy import misc
import os
from cell_classifier_tf_cnn import conv_net

# Suppress sub-optimal speed warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


# Load image, make a copy for final output, and convert image to grayscale
image_path = 'images/test_array_3_hi_res.png'
original_image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

original_image_4x = misc.imresize(original_image, (1792, 1792), interp='lanczos')
grayscale_image_4x = misc.imresize(grayscale_image, (1792, 1792), interp='lanczos')

output = original_image_4x.copy()

tf.reset_default_graph()

# Convolutional filter depths:
depths = [32, 64, 128]

# Weight and bias variables
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[3, 3, 1, depths[0]])),
    'wc2': tf.Variable(tf.random_normal(shape=[3, 3, depths[0], depths[1]])),
    'wd1': tf.Variable(tf.random_normal(shape=[9 * depths[1], depths[2]])),
    'out': tf.Variable(tf.random_normal(shape=[depths[2], 1]))}

biases = {
    'bc1': tf.Variable(tf.zeros([depths[0]])),
    'bc2': tf.Variable(tf.zeros([depths[1]])),
    'bd1': tf.Variable(tf.zeros([depths[2]])),
    'out': tf.Variable(tf.zeros([1]))}

# Placeholders
x_input = tf.placeholder(tf.float32, shape=[None, 9, 9, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
y_pred = conv_net(x_input, weights, biases, 1.0)



# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")
    print("TensorFlow model restored.")

    # Perform circle detection for droplets
    droplets = cv2.HoughCircles(grayscale_image_4x, cv2.HOUGH_GRADIENT, 1, 12, param1=120, param2=20,
                                minRadius=65, maxRadius=88)

    # Use TF classifier and a sliding window to detect droplets
    cell_ps_lookup = np.zeros(shape=(1792, 1792))
    cell_ps_list = []
    for x in range(19, 1792 - 19):
        print(x)
        for y in range(19, 1792 - 19):
            big_cell_img = grayscale_image_4x[y - 18: y + 18, x - 18: x + 18]
            cell_img = misc.imresize(big_cell_img, (9, 9))
            scaled_img = cell_img.astype(float) / 256.0
            reshaped_img = scaled_img.reshape(1, 9, 9, 1)
            cell_likelihood = sess.run(y_pred, feed_dict={x_input: reshaped_img})
            cell_ps_lookup[y, x] = cell_likelihood
            cell_ps_list.append((cell_likelihood, x, y))

    cell_ps_list.sort()
    cell_ps_list.reverse()

    cells = []

    for p, x, y in cell_ps_list:
        if p < 0.99:
            break
        if any(p < cell_ps_lookup[y + dy, x + dx] for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        if any(distance(c[0], c[1], x, y) < 10 for c in cells):
            continue
        cells.append((x, y))

    if droplets is not None:
        droplets = np.round(droplets[0, :]).astype('int')

    valid_droplets = []
    droplet_clusters = []
    cluster_lookup = {}

    # This will group together together the droplet circles, since each droplet is made of multiple circles.
    for x, y, r in droplets:

        for i, dc in enumerate(droplet_clusters):
            if any(distance(x, y, x2, y2) < 50 for x2, y2, _ in dc):      # Are any droplet circles more than 50 away?
                cluster_lookup[(x, y, r)] = i
                droplet_clusters[i].append((x, y, r))
                break
        else:
            droplet_clusters.append([(x, y, r)])

    dc_edge = 2
    droplet_clusters = [dc for dc in droplet_clusters if
                        all((dc_edge + r < x < 1792 - dc_edge - r and dc_edge + r < y < 1792 - dc_edge - r)
                            for (x, y, r) in dc)]

    for i, dc in enumerate(droplet_clusters):
        for (x, y, r) in dc:
            valid_droplets.append((x, y, r))
            cluster_lookup[(x, y, r)] = i

    cluster_count = [0] * len(droplet_clusters)
    cluster_cells = [[] for _ in range(len(droplet_clusters))]

    cell_counter = 0

    # For each cell, we want to check if it is valid, and if so, link it to its containing droplet cluster
    for x, y in cells:
        closest_droplet_circle = tuple(min(droplets, key=lambda z: distance(x, y, z[0], z[1])))
        if closest_droplet_circle not in valid_droplets:
            continue
        i = cluster_lookup[closest_droplet_circle]
        if any(distance(x, y, d_x, d_y) < d_r + 12 for d_x, d_y, d_r in droplet_clusters[i]):
            # Cell is enclosed by cluster i
            cluster_cells[i].append((x, y))

    # Go through each droplet cluster and print the droplet boundaries (commented out), the cells, and the count
    for i, dc in enumerate(droplet_clusters):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        # Printing circle boundaries
        # for x, y, r in dc:
        #     cv2.circle(output, (x, y), r, color, 1)

        for x, y in cluster_cells[i]:
            cv2.circle(output, (x, y), 10, color, 1)

        avg_x = int(sum(x for x, _, _ in dc) / len(dc))
        avg_y = int(sum(y for _, y, _ in dc) / len(dc))
        cv2.putText(output, str(len(cluster_cells[i])), (avg_x - 15, avg_y + 15), fontFace=0, fontScale=2.0,
                    color=color, thickness=4)

    # Combine original image analyzed output on one side
    image_and_output = np.hstack([original_image_4x, output])

    # Show the original and output image side by side
    cv2.imshow("output", image_and_output)
    cv2.waitKey(0)

    # Save the original and output image
    cv2.imwrite('images/output.png', image_and_output)