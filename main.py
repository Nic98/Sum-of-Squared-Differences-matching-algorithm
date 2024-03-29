import numpy as np
import cv2
from SSD import *
from SAD import *
from NCC import *
from evaluation import *
from matplotlib import pyplot as plt
from numba import njit, prange
import time


def old_evaluate(img_disparity, img_gt):
    valid = img_gt > 0
    valid_num = np.sum(valid)

    img_disparity = img_disparity * valid

    invalid_num = np.sum(img_gt == 0)

    error = np.abs(img_disparity - img_gt)
    error_0_25 = (np.sum(error < 0.25) - invalid_num) / valid_num
    error_0_5 = (np.sum(error < 0.5) - invalid_num) / valid_num
    error_1 = (np.sum(error < 1) - invalid_num) / valid_num
    error_2 = (np.sum(error < 2) - invalid_num) / valid_num
    error_4 = (np.sum(error < 4) - invalid_num) / valid_num
    print(error_0_25, error_0_5, error_1, error_2, error_4)

    rms = np.sqrt(np.sum((img_disparity - img_gt) ** 2) / np.sum(valid))
    print(rms)


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

    '''
    Try to balance the window size to a reasonable value
    '''

    ws = 40
    print("ssd with ws ", ws)
    ssd_dis = compute_SSD(left_image, right_image, ws, ws)
    old_evaluate(ssd_dis, truth_image)
    print("-" * 20)
    print("sad with ws ", ws)
    sad_dis = compute_SAD(left_image, right_image, ws, ws)
    old_evaluate(sad_dis, truth_image)
    print("-" * 20)
    print("ncc with ws ", ws)
    ncc_disparity = Original_NCC(left_image, right_image, ws)
    old_evaluate(ncc_disparity, truth_image)
    print("-" * 20)
    print("ncc + filter with ws ", ws)
    filter_ncc_disparity = filter_match(left_image, right_image, ws)
    old_evaluate(filter_ncc_disparity, truth_image)
    print("-" * 20)
    print("ncc + smooth with ws ", ws)
    filter_smooth_ncc_disparity = filter_match_smooth(left_image, right_image, ncc_disparity, ws, 0.6)
    old_evaluate(filter_smooth_ncc_disparity, truth_image)
    print("-" * 20)
    print("zncc with ws ", ws)
    zncc_disparity = Z_NCC(left_image, right_image, ws)
    old_evaluate(zncc_disparity, truth_image)
    print("-" * 20)
    print("zncc + filter with ws ", ws)
    filter_disparity = filter_match(left_image, right_image, ws, 'zncc')
    old_evaluate(filter_disparity, truth_image)
    print("-" * 20)
    print("zncc + smooth with ws ", ws)
    filter_smooth_zncc_disparity = filter_match_smooth(left_image, right_image, zncc_disparity, ws, "zncc", 0.6)
    old_evaluate(filter_smooth_zncc_disparity, truth_image)
    '''
    Using a Gaussian kernel on the window to emphasize on the
    center pixel.
    '''
    # blur_disparity = mask_Gaussian(left_image, right_image, 10, 10)
    # evaluate(blur_disparity, truth_image)

    '''
    Compute ZSSD
    '''
    # zssd_disparity = compute_ZSSD(left_image, right_image, 40, 40)
    # evaluate(zssd_disparity, truth_image)

    '''
    Compute ZSSD with search range
    '''
    # zssd_disparity_withRange = compute_ZSSD_withRange(left_image, right_image, 40, 40)
    # evaluate(zssd_disparity_withRange, truth_image)
    '''
    Smoothing
    '''
    # ground_truth = truth_image.astype(np.int8)
    # zssd_smooth_disparity = compute_ZSSD_smooth(left_image, right_image, ground_truth, 0.2, 40, 40)
    # evaluate(zssd_smooth_disparity, truth_image)

    # plt.subplot(3, 1, 1)
    # plt.imshow(left_image, cmap='gray')
    # plt.title("Original Left")
    # plt.axis('off')

    # plt.subplot(3, 1, 2)
    # plt.imshow(right_image, cmap='gray')
    # plt.title("Original Right")
    # plt.axis('off')
    # plt.show()
    plt.subplot(3, 1, 1)
    plt.imshow(zncc_disparity, cmap='gray')
    plt.title("zncc disparity")
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(filter_disparity, cmap='gray')
    plt.title("zncc filter disparity")
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(filter_smooth_zncc_disparity, cmap='gray')
    plt.title("disparity")
    plt.axis('off')
    plt.show()
    pass
