import numpy as np
import cv2
from SSD import *
from SAD import *
from NCC import *
from evaluation import *
from matplotlib import pyplot as plt
from numba import njit, prange
import time

if __name__ == '__main__':
    # Ground Truth image
    truth_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-disparity.png', -1) / 256
    # Image Left
    left_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-left.jpg')
    # Image Right
    right_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-right.jpg')
    # Read the images in Grayscale format
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY).astype(np.int32)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY).astype(np.int32)

    ws = 10

    zncc_disparity = Z_NCC(left_image, right_image, ws) # 22.19
    e1, e2, e3, e4, e5, rms = evaluate(zncc_disparity, truth_image)
    print("ZNCC WS10 rms = ", rms)
    smooth_zncc_disparity = disparity_smooth(left_image, right_image, zncc_disparity, ws, "zncc", 0.6) # 18.49
    e1, e2, e3, e4, e5, rms = evaluate(smooth_zncc_disparity, truth_image)
    print("ZNCC + smoothing WS10 rms = ", rms)
    plt.subplot(2, 1, 1)
    plt.imshow(zncc_disparity, cmap='gray')
    plt.title("zncc disparity")
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(smooth_zncc_disparity, cmap='gray')
    plt.title("zncc smooth disparity")
    plt.axis('off')
    plt.show()
    pass