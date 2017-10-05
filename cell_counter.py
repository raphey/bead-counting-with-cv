__author__ = 'raphey'

import numpy as np
import cv2


# Specify image path
image_path = 'images/test_array_lo_res.png'


image = cv2.imread(image_path)
output = image.copy()
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

droplets = cv2.HoughCircles(image=grayscale_image, method=cv2.HOUGH_GRADIENT, dp=1,
                            minDist=20, param1=80, param2=1, minRadius=11, maxRadius=11)

cells = cv2.HoughCircles(image=grayscale_image, method=cv2.HOUGH_GRADIENT, dp=1,
                         minDist=4, param1=80, param2=2, minRadius=0, maxRadius=3)


if len(droplets) > 0:
    # convert the (x, y) coordinates and radius of the droplets and cells to integers
    droplets = np.round(droplets[0, :]).astype('int')
    cells = np.round(cells[0, :]).astype('int')


    for x, y, r in droplets:
        print(x, y, r)
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 0, 255), 1)
        cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

    for x, y, r in cells:
        print(x, y, r)
        # draw the circle in the output image
        cv2.circle(output, (x, y), r, (0, 255, 0), 1)

    image_and_output = np.hstack([image, output])

    # show the original and output image
    cv2.imshow("output", image_and_output)
    cv2.waitKey(0)

    # save the original and output image
    cv2.imwrite('images/output.png', image_and_output)