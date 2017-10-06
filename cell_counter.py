__author__ = 'raphey'

import numpy as np
import cv2
from simple_cell_classifier import fwd_pass


def cell_filter(img, x_c, y_c, window=0, threshold=0.8):
    """
    Uses a linear classifier to determine of a 28x28 image centered at x_c, y_c is a cell.
    Slides a window around to look for maximum likelihood; also stores the maximum center in a
    global list after checking that it isn't too close to an already-found center.
    """
    if not (14 + window <= x_c <= len(img[0]) - 14 - window and 14 + window <= y_c <= len(img) - 14 - window):
        return False

    max_likelihood = 0.0
    best_x, best_y = 0, 0

    for x_shift in range(-window, window + 1):
        for y_shift in range(-window, window + 1):
            x_min = x_c - 14 + x_shift
            y_min = y_c - 14 + y_shift
            cell_img = img[y_min: y_min + 28, x_min: x_min + 28].astype(float)
            min_val = cell_img.min()
            max_val = cell_img.max()
            scaled_img = (cell_img - min_val) / (max_val - min_val)
            reshaped_img = scaled_img.reshape(1, 784)
            cell_likelihood = fwd_pass(reshaped_img, weight, bias)

            if cell_likelihood > max_likelihood:
                max_likelihood = cell_likelihood
                best_x, best_y = x_c + x_shift, y_c + y_shift

    if max_likelihood > threshold:
        if all((best_x - x)**2 + (best_y - y)**2 > 49.0 for x, y in filtered_cells):
            filtered_cells.append((best_x, best_y))
            return True
    else:
        return False




def save_cell(img, x_c, y_c, a=14):
    if not (a <= x_c <= len(img[0]) - a and a <= y_c <= len(img) - a):
        return
    cell_img = img[y_c - a: y_c + a, x_c - a: x_c + a]
    if len(cell_img[0]) < 2 * a:
        print(x_c, y_c)
        print(len(cell_img), len(cell_img[0]))
        quit()
    count_label = str(100000 + counter)[1:]
    cv2.imwrite('training_data/sample_{}_{}_x{}_y{}.png'.format(2 * a, count_label, x_c, y_c), cell_img)


# Specify image path
image_path = 'images/test_array_hi_res_4x.png'

image = cv2.imread(image_path)
output = image.copy()
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform circle detection for droplets and cells
droplets =cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 12, param1=80, param2=30, minRadius=65, maxRadius=93)
# Used for getting training data:
cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 8, param1=40, param2=4, minRadius=10, maxRadius=12)

# Alternate cell detection allowing for tiny white circles in cell centers
# cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 8, param1=40, param2=4, minRadius=3, maxRadius=12)


filtered_cells = []

# Load weight and bias for linear classifier
weight = np.load('classifier_data/weight.npy')
bias = np.load('classifier_data/bias.npy')

if len(droplets) > 0:
    # convert the (x, y) coordinates and radius of the droplets and cells to integers
    droplets = np.round(droplets[0, :]).astype('int')
    cells = np.round(cells[0, :]).astype('int')


    for x, y, r in droplets:
        # print(x, y, r)
        # draw the droplet circle in the output image, then draw a small square at the center
        cv2.circle(output, (x, y), r, (0, 0, 255), 1)
        cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

    counter = 0
    for x, y, r in cells:
        # print(x, y, r)
        if cell_filter(grayscale_image, x, y, window=9):
            # draw the circle in the output image
            counter += 1
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)

            # Save image to file
            # save_cell(grayscale_image, x, y)

    print("Total number of cells detected:", counter)

    image_and_output = np.hstack([image, output])

    # show the original and output image
    cv2.imshow("output", image_and_output)
    cv2.waitKey(0)

    # save the original and output image
    cv2.imwrite('images/output.png', image_and_output)