import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

filename = "../orthomosaic.tif"
count = []
with rasterio.open(filename) as src:
    columns = src.width
    rows = src.height
    ulc = [0, 0]
    lrc = [1000, 1000]
    for i in range(23):
        if i == 0:
            window_location = Window.from_slices(
                (ulc[0], lrc[0]),
                (ulc[1], lrc[1]))
        else:
            window_location = Window.from_slices(
                (ulc[0] + (i * rows / 23), lrc[0] + (i * rows / 23)),
                (ulc[1] + (i * columns / 23), lrc[1] + (i * columns / 23)))
        img = src.read(window=window_location)
        temp = img.transpose(1, 2, 0)
        t2 = cv2.split(temp)
        img_cv = cv2.merge([t2[2], t2[1], t2[0]])
        # cv2.imshow('image', img_cv)
        # cv2.imwrite('image'+str(i)+'.jpg', img_cv)
        # cv2.waitKey(0)
        # image mask
        # img_annot = cv2.imread('image_mask.jpg')
        img_annot = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        lower_limit = (0, 0, 101)
        upper_limit = (40, 100, 255)
        mask = cv2.inRange(img_cv, lower_limit, upper_limit)
        # cv2.imwrite('red_mask.jpg', mask)
        # mean, std = cv2.meanStdDev(img_cv, mask=mask)
        # calculate the pixel values into a list
        pixels = np.reshape(img_cv, (-1, 3))
        mask_pixels = np.reshape(mask, (-1))
        annot_pix_values = pixels[mask_pixels == 255, ]
        avg = np.average(annot_pix_values, axis=0)  # centroid location
        cov = np.cov(annot_pix_values.transpose())
        # calculate mahalanobis distance
        shape = pixels.shape
        diff = pixels - np.repeat([avg], shape[0], axis=0)
        temp = np.linalg.inv(cov)
        mahalanobis_dist = np.sum(diff * (diff @ temp), axis=1)
        mahalanobis_distance_image = np.reshape(mahalanobis_dist, (img_cv.shape[0],
                                                                   img_cv.shape[1]))
        mahal_scaled_dist_image = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
        cv2.imwrite("orthomosaic_mahalanobis_dist_image.jpg", mahal_scaled_dist_image)

        _, mahalanobis_segmented = cv2.threshold(mahalanobis_distance_image, 5, 255,
                                                 cv2.THRESH_BINARY_INV)
        mahalanobis_segmented = mahalanobis_segmented.astype(np.uint8)
        cv2.imwrite("orthomosaic_mahalanobis_dist_segmented.jpg",
                    mahalanobis_segmented)
        # finding edges of the orange blobs
        blobs = cv2.imread("orthomosaic_mahalanobis_dist_segmented.jpg")
        # applying median filter
        img = cv2.medianBlur(blobs, 3)
        # finding edges of the orange blobs
        edged = cv2.Canny(img, 101, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # amount of contours in filtered image
        count.append(len(contours))
        print(len(contours))
        cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 3)
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", img_cv)
        cv2.waitKey(0)

total = 0
for x in count:
    total += x

print("pumpkin total is "+str(total))
