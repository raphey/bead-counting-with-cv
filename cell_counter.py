__author__ = 'raphey'

import numpy as np
import cv2


def simple_cell_filter(img, x_c, y_c, a=5):
    """
    The idea here was to do a simple filter to cut down on the false positives for cells.
    Turns out it's hard to get this to do too much good.
    """
    # Filter is turned off...
    return True

    x_min = max(0, x_c - a)
    x_max = min(len(img[0]) - 1, x_c + a + 1)
    y_min = max(0, y_c - a)
    y_max = min(len(img) - 1, y_c + a + 1)
    sub_array = img[y_min: y_max, x_min: x_max]  # Tricky/annoying! Indices switched
    avg_value = int(sub_array.mean())
    min_value = sub_array.min()
    max_value = sub_array.max()
    std_dev = int(sub_array.std())
    c = img[y_c][x_c]
    up = img[y_min][x_c]
    down = img[y_max][x_c]
    left = img[y_c][x_min]
    right = img[y_c][x_max]
    if min_value < 50:   # Too dark
        return False

    if max_value - min_value < 25:  # Too uniform
        return False

    return sum(c > d for d in [up, down, left,right]) >= 3



# Specify image path
# image_path = 'images/test_array_lo_res.png'
# image_path = 'images/test_array_hi_res_2x_edge_threshold2.png'
# image_path = 'images/huge_and_threshold.png'
image_path = 'images/test_array_hi_res_4x.png'

image = cv2.imread(image_path)
output = image.copy()
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# droplets = cv2.HoughCircles(image=grayscale_image, method=cv2.HOUGH_GRADIENT, dp=1,
#                             minDist=20, param1=80, param2=1, minRadius=11, maxRadius=11)
# cells = cv2.HoughCircles(image=grayscale_image, method=cv2.HOUGH_GRADIENT, dp=1,
#                          minDist=4, param1=80, param2=2, minRadius=0, maxRadius=3)

# Hi res circles big
# droplets =cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 80, param1=80, param2=3, minRadius=40, maxRadius=48)
# Hi res tiny-guys big
# cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 7, param1=20, param2=3, minRadius=4, maxRadius=7)

# Hi res circles huge, thresholded
# droplets =cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 160, param1=80, param2=3, minRadius=80, maxRadius=96)
# Hi res tiny-guys huge, thresholded
# cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 14, param1=100, param2=10, minRadius=3, maxRadius=12)

# Hi res circles huge, default scaling. Purposely getting multiple circles for each droplet
droplets =cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 12, param1=80, param2=30, minRadius=65, maxRadius=93)

# Hi res tiny-guys huge, default scaling. Getting way more cells than we want, with the goal of filtering later.
# cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 8, param1=30, param2=5, minRadius=7, maxRadius=11)

# More conservative, getting some false negatives
cells = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 8, param1=30, param2=10, minRadius=7, maxRadius=11)



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
        if simple_cell_filter(grayscale_image, x, y):
            # draw the circle in the output image
            counter += 1
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)

    print("Total number of cells detected:", counter)

    image_and_output = np.hstack([image, output])

    # show the original and output image
    cv2.imshow("output", image_and_output)
    cv2.waitKey(0)

    # save the original and output image
    cv2.imwrite('images/output.png', image_and_output)