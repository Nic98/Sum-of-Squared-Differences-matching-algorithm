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
    truth_image3 = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-55-15-689-disparity.png', -1) / 256
    # Image Left
    left_image3 = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-55-15-689-left.jpg')
    # Image Right
    right_image3 = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-55-15-689-right.jpg')
    # Read the images in Grayscale format
    left_image3 = cv2.cvtColor(left_image3, cv2.COLOR_RGB2GRAY).astype(np.int32)
    right_image3 = cv2.cvtColor(right_image3, cv2.COLOR_RGB2GRAY).astype(np.int32)
    windows = [10, 30, 50, 70]
    ws = 10

    zncc_disparity10 = Z_NCC(left_image3, right_image3, ws)  # 22.19
    e1, e2, e3, e4, e5, rms = evaluate(zncc_disparity10, truth_image3)
    print("ZNCC WS10 rms = ", rms)
    zncc_disparity30 = Z_NCC(left_image3, right_image3, 30)  # 22.19
    e1, e2, e3, e4, e5, rms = evaluate(zncc_disparity30, truth_image3)
    print("ZNCC WS30 rms = ", rms)
    plt.subplot(3, 1, 1)
    plt.imshow(truth_image3, cmap='gray')
    plt.title("ground truth disparity")
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(zncc_disparity10, cmap='gray')
    plt.title("zncc ws10 disparity")
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(zncc_disparity30, cmap='gray')
    plt.title("zncc ws30 disparity")
    plt.axis('off')
    plt.show()
    pass