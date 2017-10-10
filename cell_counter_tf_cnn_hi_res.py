__author__ = 'raphey'


from cell_counter_tf_cnn import *

import time

# Overwrites function being imported from cell_counter_tf_cnn
def find_cells(image_4x):
    """
    Higher resolution cell detector.
    """

    # Create variables for CNN model import
    weights, biases, x_input, y_actual, y_pred = tf_model_setup()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore variables from disk.
        saver.restore(sess, "classifier_data/tf_cnn_classifier/tf_cnn_model.ckpt")
        print("TensorFlow model restored.")

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
            print(i)
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


if __name__ == '__main__':
    # Suppress sub-optimal speed warnings from TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Random seed for consistent color display on multiple runs
    np.random.seed(0)

    _, grayscale_image_4x, original_image_4x = read_image('images/test_array_3_hi_res.png')

    droplet_data = find_and_process_droplets(grayscale_image_4x)

    cells = find_cells(grayscale_image_4x)

    cells_by_cluster = group_cells_by_cluster(cells, droplet_data)

    write_output(original_image_4x, droplet_data, cells_by_cluster, display=False, save=True)

