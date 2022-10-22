import numpy as np
import cv2
from SSD import *
from SAD import *
from NCC import *
from evaluation import *
from matplotlib import pyplot as plt

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

    ws = 50
    # SAD
    print("{} with window size {} ".format("SAD", ws))
    sad_dis = compute_SAD(left_image, right_image, ws, ws)
    e1, e2, e3, e4, e5, rms = evaluate(sad_dis, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("SAD", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # SSD
    print("{} with window size {} ".format("SSD", ws))
    ssd_dis = ssd(left_image, right_image, ws, ws)
    e1, e2, e3, e4, e5, rms = evaluate(ssd_dis, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("SSD", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # NCC
    print("{} with window size {} ".format("NCC", ws))
    ncc_disparity = NCC(left_image, right_image, ws)
    e1, e2, e3, e4, e5, rms = evaluate(ncc_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("NCC", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # # NCC+Kernel
    print("{} with window size {} ".format("NCC + Weight Kernel", ws))
    filter_ncc_disparity = filter_match(left_image, right_image, ws)
    e1, e2, e3, e4, e5, rms = evaluate(filter_ncc_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("NCC+Kernel", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # # NCC+Smoothing
    print("{} with window size {} ".format("NCC + smoothing", ws))
    smooth_ncc_disparity = disparity_smooth(left_image, right_image, ncc_disparity, ws, 0.6)
    e1, e2, e3, e4, e5, rms = evaluate(smooth_ncc_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("NCC+smoothing", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # ZNCC
    print("{} with window size {} ".format("ZNCC", ws))
    zncc_disparity = ZNCC(left_image, right_image, ws)
    e1, e2, e3, e4, e5, rms = evaluate(zncc_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("ZNCC", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # ZNCC+Kernel
    print("{} with window size {} ".format("ZNCC + Weight Kernel", ws))
    zncc_filter_disparity = filter_match(left_image, right_image, ws, 'zncc')
    e1, e2, e3, e4, e5, rms = evaluate(zncc_filter_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("ZNCC+Kernel", e1, e2, e3, e4, e5, rms))
    print("-" * 20)
    # ZNCC+Smoothing
    print("{} with window size {} ".format("ZNCC + smoothing", ws))
    smooth_zncc_disparity = disparity_smooth(left_image, right_image, zncc_disparity, ws, "zncc", 0.6)
    e1, e2, e3, e4, e5, rms = evaluate(smooth_zncc_disparity, truth_image)
    print("{}: error 0.25 = {:.3f}, error 0.5 = {:.3f}, error 1 = {:.3f}, error 2 = {:.3f}, error 4 = {:.3f}, rms = {:.3f}".format("ZNCC+smoothing", e1, e2, e3, e4, e5, rms))

    plt.subplot(3, 1, 1)
    plt.imshow(zncc_disparity, cmap='gray')
    plt.title("zncc disparity")
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(zncc_filter_disparity, cmap='gray')
    plt.title("zncc filter disparity")
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(smooth_zncc_disparity, cmap='gray')
    plt.title("smooth zncc disparity")
    plt.axis('off')
    plt.show()
    pass