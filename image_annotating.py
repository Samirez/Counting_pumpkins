import cv2
import numpy as np
from matplotlib import pyplot as plt

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[: -1][f == np.max(f)])
clip = lambda z: np.maximum(1e-30, z)


def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)


def preliminaries(n, x):
    """Some math that is shared across each algorithm."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[: -1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x ** 2) - w0 * mu0 ** 2
    d1 = dsum(n * x ** 2) - w1 * mu1 ** 2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


# generalized histogram thresholding
def GHT(n, x=None, nu=0, tau=0, kappa=0.0, omega=0.5):
    """Our generalization of the above algorithms."""
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert 0 <= omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0 * nu * tau ** 2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau ** 2 + d1) / (p1 * nu + w1))
    f0 = - d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = - d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1


# Load image and segment it with a mask
# image = cv2.imread('./2019-03-19 Images for third miniproject/EB-02-660_0595_0007.jpg')
image = cv2.imread('./2019-03-19 Images for third miniproject/EB-02-660_0594_0326.jpg')
img_annot = cv2.imread('image_mask.jpg')
lower_limit = (0, 0, 200)
upper_limit = (100, 100, 255)
# finding orange nuance of pumpkins with color mask
'''ORANGE_MIN = np.array([5, 50, 50], np.uint8)
ORANGE_MAX = np.array([15, 255, 255], np.uint8)
COLOR_MIN = ORANGE_MIN
COLOR_MAX = ORANGE_MAX'''
# Adding tge image mask and calculating the mean and standard deviation
mask = cv2.inRange(img_annot, lower_limit, upper_limit)
cv2.imwrite('red_mask.jpg', mask)
mean, std = cv2.meanStdDev(image, mask=mask)
print(mean, std)
# calculate the pixel values into a list
pixels = np.reshape(image, (-1, 3))
mask_pixels = np.reshape(mask, (-1))
annot_pix_values = pixels[mask_pixels == 255,]
avg = np.average(annot_pix_values, axis=0)  # centroid location
cov = np.cov(annot_pix_values.transpose())
# storing the pixel values in histogram
print('mean is ', avg)

# calculate mahalanobis distance
shape = pixels.shape
diff = pixels - np.repeat([avg], shape[0], axis=0)
temp = np.linalg.inv(cov)
mahalanobis_dist = np.sum(diff * (diff @ temp), axis=1)
mahalanobis_distance_image = np.reshape(mahalanobis_dist, (image.shape[0],
                                                           image.shape[1]))
mahal_scaled_dist_image = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
cv2.imwrite("mahalanobis_dist_image.jpg", mahal_scaled_dist_image)

_, mahalanobis_segmented = cv2.threshold(mahalanobis_distance_image, 5, 255,
                                         cv2.THRESH_BINARY_INV)
mahalanobis_segmented = mahalanobis_segmented.astype(np.uint8)
cv2.imwrite("mahalanobis_dist_segmented.jpg",
            mahalanobis_segmented)
# determine threshold for segmenting image
hist_n, hist_edge = np.histogram(mahal_scaled_dist_image, np.arange(-0.5, 256))
hist_x = (hist_edge[1:] + hist_edge[:-1]) / 2.
threshold_GHT, valb = GHT(hist_n, hist_x, nu=2 ** 5, tau=2 ** 10, kappa=0.1, omega=0.5)
print("Threshold value found by Generalized Histogram Thresholding")
print(threshold_GHT)
thresholded_image = (mahal_scaled_dist_image < threshold_GHT) * 255
cv2.imwrite("generalized_histogram_thresholding.jpg", thresholded_image)

'''np.savetxt("annotated_pixel_values.csv",
           annot_pix_values,
           delimiter=",",
           fmt="%d")'''
# visualizing the distribution
'''fig, ax11 = plt.subplots()
ax11.plot(pixels[:, 1], pixels[:, 2], '.')
ax11.set_title('Color values of pumpkins')
plt.xlabel("Green [0 - 255]")
plt.ylabel("Red [0 - 255]")
fig.tight_layout()
# plt.savefig("color_distribution.pdf", dpi=150)
plt.plot()
plt.show()'''
