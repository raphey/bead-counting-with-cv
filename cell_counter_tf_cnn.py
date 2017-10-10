__author__ = 'raphey'

import numpy as np
import cv2
import tensorflow as tf
from scipy import misc
import os
from cell_classifier_tf_cnn import conv_net
import csv
import glob


def read_image(image_path):
    """
    Reads an image from file and returns a grayscale version, an interpolated 4x
    grayscale version, and an interpolated 4x original version.
    """
    original = cv2.imread(image_path)
    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayscale_4x = misc.imresize(grayscale, (1792, 1792), interp='lanczos')
    original_4x = misc.imresize(original, (1792, 1792), interp='lanczos')

    return grayscale, grayscale_4x, original_4x


def find_and_process_droplets(image_4x):
    """
    Given a grayscale 1792x1792 image, returns a data structure with detected droplets.

    The process is:
    1) Detect droplet-sized circles using OpenCVs HoughCircles, with multiple circles
       per actual droplet so as to match the irregular shapes better
    2) Group the circles by proximity; these are droplet clusters.
    3) Throw out droplet clusters that are too close to the edge

    The data structure returned has the following components:

    - droplets: An (x, y, r) list of all droplet-sized circles detected
    - valid_droplets: A smaller (x, y, r) list of of those that aren't too
      close to the edge or have associations with circles that are too close
      to the edge.
    - droplet_clusters: A list of lists of (x, y, r) circles, grouped by
      proximity
    - cluster_lookup: A dictionary in which each (x, y, r) key contains the
      the cluster index position in droplet_clusters for a given droplet.
    """
    # Perform circle detection for droplets
    ds = cv2.HoughCircles(image_4x, cv2.HOUGH_GRADIENT, 1, minDist=12, param1=120,
                          param2=20, minRadius=65, maxRadius=88)

    if ds is None:
        raise ValueError('No droplets detected with current image and parameters')

    ds = np.round(ds[0, :]).astype('int')

    valid_ds = []
    d_clusters = []
    c_lookup = {}

    # Group together together droplet circles, since each droplet is made of multiple circles.
    for x, y, r in ds:

        for i, d_cluster in enumerate(d_clusters):
            if any(distance(x, y, x2, y2) < 50 for x2, y2, _ in d_cluster):
                c_lookup[(x, y, r)] = i
                d_clusters[i].append((x, y, r))
                break
        else:
            d_clusters.append([(x, y, r)])

    # Avoid using droplets that are within a certain number of pixels from the edge
    edge = 2

    # Filter down to only clusters that contain all valid droplets
    d_clusters = [dc for dc in d_clusters if all((edge + r < x < 1792 - edge - r and
                                                  edge + r < y < 1792 - edge - r)
                                                 for (x, y, r) in dc)]

    # Make list of valid droplets and look-up dictionary for cluster assignment
    for i, d_cluster in enumerate(d_clusters):
        for (x, y, r) in d_cluster:
            valid_ds.append((x, y, r))
            c_lookup[(x, y, r)] = i

    return {'droplets': ds, 'valid_droplets': valid_ds,
            'droplet_clusters': d_clusters, 'cluster_lookup': c_lookup}


def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def find_cells(image):
    """
    Given a grayscale 448x448 image, uses a trained CNN to get a list of cells.

    The process is:
    1) Pass a 9x9 window across all image locations, minus some padding, and
       for each one, find an associated probability of it being a cell.
    2) Go through the entire list of probabilities in decreasing order. Mark
       a cell as found if it has probability above a certain cutoff, it is a
       local maximum of probability, and it is a minimum distance away from
       other cells that have already been found

    Returns a list of detected cells in (x, y) form. Note that these coordinates
    are scaled up 4x from the image being passed in
    """

    # Create variables for CNN model import
    weights, biases, x_input, y_actual, y_pred = tf_model_setup()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore variables from disk.
        saver.restore(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")

        # Use TF classifier and a sliding window to detect droplets
        img_tensor = []
        for x in range(4, 444):
            for y in range(4, 444):
                cell_img = image[y - 4: y + 5, x - 4: x + 5].astype(float)
                scaled_img = cell_img / 256.0
                reshaped_img = scaled_img.reshape(9, 9, 1)
                img_tensor.append(reshaped_img)

        tf_model_output = sess.run(y_pred, feed_dict={x_input: np.array(img_tensor)}).reshape(440, 440, 1)

        c_probs_lookup = np.zeros(shape=(448, 448))

        c_probs = []

        for x in range(440):
            for y in range(440):
                p = tf_model_output[x, y, 0]
                c_probs.append((p, x + 4, y + 4))
                c_probs_lookup[y + 4, x + 4] = p

        c_probs.sort()
        c_probs.reverse()

        # Grab valid cells
        cs = []
        for p, x, y in c_probs:
            if p < 0.99:
                break
            if any(p < c_probs_lookup[y + dy, x + dx] for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            if any(distance(c[0], c[1], 4 * x, 4 * y) < 10 for c in cs):
                continue
            cs.append((4 * x, 4 * y))

        return cs


def hi_res_find_cells(image_4x):
    """
    Higher resolution cell detector.
    """

    # Create variables for CNN model import
    weights, biases, x_input, y_actual, y_pred = tf_model_setup()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore variables from disk.
        saver.restore(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")

        downsampled_imgs = [[None] * 4 for _ in range(4)]

        for x in range(4):
            for y in range(4):
                shift_img = misc.imresize(image_4x[y: y - 4, x: x - 4], (447, 447))
                shift_img = shift_img.astype(float) / 256.0
                shift_img = shift_img.reshape(447, 447, 1)
                downsampled_imgs[y][x] = shift_img

        img_list = []
        for x in range(19, 1792 - 20):
            for y in range(19, 1792 - 20):

                shift_x = x % 4
                shift_y = y % 4
                scaled_x = x // 4
                scaled_y = y // 4

                cell_image = downsampled_imgs[shift_y][shift_x][scaled_y - 4: scaled_y + 5, scaled_x - 4: scaled_x + 5]

                img_list.append(cell_image)

        img_tensor = np.array(img_list)

        print("Image tensor formed.")

        batch_size = 200000

        tf_model_output = np.empty((0, 1))

        for i in range(0, len(img_tensor), batch_size):
            batch_output = sess.run(y_pred, feed_dict={x_input: np.array(img_tensor[i:i + batch_size])})
            tf_model_output = np.vstack((tf_model_output, batch_output))

        print("Probabilities computed.")

        tf_model_output = tf_model_output.reshape(1753, 1753, 1)

        c_probs_lookup = np.zeros(shape=(1792, 1792))

        c_probs = []

        for x in range(1753):
            for y in range(1753):
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


def tf_model_setup():
    """
    Prepare variables for import of trained TensorFlow classifier
    """

    tf.reset_default_graph()

    # Convolutional filter depths:
    depths = [32, 64, 128]

    # Weight and bias variables (to be filled with values when model loads)
    weights = {
        'wc1': tf.Variable(tf.zeros(shape=[3, 3, 1, depths[0]])),
        'wc2': tf.Variable(tf.zeros(shape=[3, 3, depths[0], depths[1]])),
        'wd1': tf.Variable(tf.zeros(shape=[9 * depths[1], depths[2]])),
        'out': tf.Variable(tf.zeros(shape=[depths[2], 1]))}

    biases = {
        'bc1': tf.Variable(tf.zeros([depths[0]])),
        'bc2': tf.Variable(tf.zeros([depths[1]])),
        'bd1': tf.Variable(tf.zeros([depths[2]])),
        'out': tf.Variable(tf.zeros([1]))}

    # Placeholders
    x_input = tf.placeholder(tf.float32, shape=[None, 9, 9, 1])
    y_actual = tf.placeholder(tf.float32, shape=[None, 1])
    y_pred = conv_net(x_input, weights, biases, 1.0)

    return weights, biases, x_input, y_actual, y_pred


def group_cells_by_cluster(cs, d_data):
    """
    Given a list of detected cells and droplet data, returns a list of cells grouped
    by the cluster they belong to, if they belong to a valid cluster.
    The assumption is that a cell belongs to the cluster of whatever cluster circle
    has the closest center, as long as it is contained by some circle in that cluster.
    """
    ds = d_data['droplets']
    valid_ds = d_data['valid_droplets']
    d_clusters = d_data['droplet_clusters']
    cluster_lookup = d_data['cluster_lookup']

    cs_by_cluster = [[] for _ in range(len(d_clusters))]

    # Wiggle room for cell containment--can lead to some false positives on the edges
    edge_error = 12

    # For each cell, check if valid, and if so, link it to its containing droplet cluster
    for x, y in cs:
        closest_d = tuple(min(ds, key=lambda z: distance(x, y, z[0], z[1])))
        if closest_d not in valid_ds:
            continue
        i = cluster_lookup[closest_d]
        if any(distance(x, y, d_x, d_y) < d_r + edge_error for d_x, d_y, d_r in d_clusters[i]):
            # Cell is enclosed by cluster i
            cs_by_cluster[i].append((x, y))

    return cs_by_cluster


def write_image_output(orig_img_4x, d_data, cluster_cs, display_image=True, show_droplets=False,
                       save_image=False, save_path='images/output.png'):
    """
    Displays and/or writes cell and droplet info to disk.
    """

    output = orig_img_4x.copy()

    # Go through each droplet cluster and print the droplet boundaries (commented out), the cells, and the count
    for i, dc in enumerate(d_data['droplet_clusters']):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        if show_droplets:
            # Printing circle boundaries
            for x, y, r in dc:
                cv2.circle(output, (x, y), r, color, 1)

        for x, y in cluster_cs[i]:
            cv2.circle(output, (x, y), 10, color, 1)

        avg_x = int(sum(x for x, _, _ in dc) / len(dc))
        avg_y = int(sum(y for _, y, _ in dc) / len(dc))
        cv2.putText(output, str(len(cluster_cs[i])), (avg_x - 15, avg_y + 15), fontFace=0, fontScale=2.0,
                    color=color, thickness=4)

    # Combine original image and analyzed output side by side
    image_and_output = np.hstack([orig_img_4x, output])

    if display_image:
        # Show the original and output image side by side
        cv2.imshow("output", image_and_output)
        cv2.waitKey(0)

    if save_image:
        cv2.imwrite(save_path, image_and_output)


def get_frequencies(cs_by_cluster):
    """

    """
    cell_counts = [len(cs) for cs in cs_by_cluster]
    cell_counts.sort()
    max_count = cell_counts[-1]
    fs = [0] * (max_count + 1)
    for count in cell_counts:
        fs[count] += 1
    return list(zip(range(max_count + 1), fs))


def analyze_image(img_path, hi_res=False, display_image=False, show_droplets=False,
                  save_image=False, save_frequencies=True):
    """
    Given an image path, detects droplets, detects cells, associates cells with
    droplets, saves/displays image, saves/displays frequencies (cells per droplet).
    """

    image_name = os.path.basename(os.path.normpath(img_path))
    print('-' * 20)
    print("Analyzing {}...".format(img_path))

    grayscale_image, grayscale_image_4x, original_image_4x = read_image(img_path)

    droplet_data = find_and_process_droplets(grayscale_image_4x)

    if hi_res:
        cells = hi_res_find_cells(grayscale_image_4x)
    else:
        cells = find_cells(grayscale_image)

    cells_by_cluster = group_cells_by_cluster(cells, droplet_data)

    write_image_output(original_image_4x, droplet_data, cells_by_cluster,
                       display_image=display_image, show_droplets=show_droplets,
                       save_image=save_image)

    frequencies = get_frequencies(cells_by_cluster)

    print("Cell per droplet frequencies for {}:".format(image_name))

    for count, freq in frequencies:
        print("{}:\t{}".format(count, freq))

    if save_frequencies:
        if hi_res:
            freq_save_path = 'frequency_output/' + image_name[:-4] + '_hi_res.csv'
        else:
            freq_save_path = 'frequency_output/' + image_name[:-4] + '.csv'
        with open(freq_save_path, 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['Count', 'Frequency'])
            for row in frequencies:
                csv_out.writerow(row)


def analyze_directory(directory_path, hi_res=False):
    for image_path in glob.glob(directory_path + '/*.png'):
        analyze_image(image_path, hi_res=hi_res)


if __name__ == '__main__':

    # Suppress sub-optimal speed warnings from TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Random seed for consistent color display on multiple runs
    np.random.seed(0)

    # analyze_image(img_path='images/test_array_2.png')

    analyze_directory('images/counter_validation_images', hi_res=True)