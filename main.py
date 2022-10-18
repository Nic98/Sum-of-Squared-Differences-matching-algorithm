import numpy as np
import cv2
from SSD import*
from matplotlib import pyplot as plt
from numba import njit, prange

if __name__ == '__main__':
    # Ground Truth image
    truth_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-disparity.png', -1) / 256
    # Image Left
    left_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-left.jpg')
    # Image Right
    right_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-right.jpg')
    # Read the images in Grayscale format
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY).astype(np.int64)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY).astype(np.int64)

    compute_SSD(left_image, right_image, 1)

    plt.subplot(2, 1, 1)
    plt.imshow(left_image, cmap='gray')
    plt.title("Original Left")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(right_image, cmap='gray')
    plt.title("Original Right")
    plt.axis('off')
    plt.show()


    pass

