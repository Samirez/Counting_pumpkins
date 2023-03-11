import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_orig = cv.imread('./2019-03-19 Images for third miniproject/EB-02-660_0594_0326.jpg')
'''
ORANGE_MIN = np.array([5, 50, 101], np.uint8)
ORANGE_MAX = np.array([20, 80, 255], np.uint8)
# segmenting image with orange color threshold
blobs = cv.inRange(img, ORANGE_MIN, ORANGE_MAX)'''
# finding edges of the orange blobs
blobs = cv.imread("mahalanobis_dist_segmented.jpg")
edged = cv.Canny(blobs, 101, 200)
contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# amount of contours in segmented image
print(str(len(contours)))
# applying median filter
img = cv.medianBlur(blobs, 3)
# blobs = cv.inRange(img, ORANGE_MIN, ORANGE_MAX)
# finding edges of the orange blobs
edged = cv.Canny(img, 101, 200)
contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# amount of contours in filtered image
print(len(contours))
# drawing contours on image
'''for i, c in enumerate(contours):
    # create a blank image of same shape
    black = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
    # draw a single contour as a mask
    single_object_mask = cv.drawContours(black, [c], 0, 255, -1)
    # coordinates containing white pixels of mask
    coords = np.where(single_object_mask == 255)
    # pixel intensities present within the image at these locations
    pixels = img[coords]
    # plot histogram
    plt.hist(pixels, 255, [0, 255])
    plt.savefig('pumpkin_histogram.jpg'.format(i))
    # plt.clf()'''
cv.drawContours(img_orig, contours, -1, (0, 255, 0), 3)
cv.namedWindow("contours", cv.WINDOW_NORMAL)
cv.imshow("contours", img_orig)
cv.waitKey(0)
