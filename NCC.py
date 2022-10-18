import numpy as np
from math import floor
from copy import deepcopy
from numba import njit, prange


@njit(parallel=True)
def Original_NCC(left_img, right_img, window_size=9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    print("height: ", height)
    print("width: ", width)
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)

    for left_y in prange(height):
        for left_x in prange(width):
            wleft, wright, wdown, wup = threshold, threshold, threshold, threshold
            if left_x < threshold:
                wleft = left_x
            if left_y < threshold:
                wup = left_y
            if left_x >= width - threshold:
                wright = width - left_x - 1
            if left_y > height - threshold:
                wdown = height - left_y - 1
            window_width = (left_x - wleft, left_x + wright + 1)
            window_height = (left_y - wup, left_y + wdown + 1)
            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]), int(window_width[0]):int(window_width[1])])
            best_match_disparity = 999999999
            score = 0
            for right_x in prange(left_x + 1):
                if right_x - wleft < 0:
                    pass
                elif right_x + wright + 1 > width:
                    pass
                else:
                    right_window_width = (int(right_x - wleft), int(right_x + wright + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1], right_window_width[0]:right_window_width[1]])
                    n = (window_width[1] - window_width[0]) * (window_height[1] - window_height[0])
                    ncc_score = np.sum(np.multiply(left_window, right_window))
                    ncc_score = ncc_score / (np.sqrt(np.sum(left_window**2)) * np.sqrt(np.sum(right_window**2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx

@njit(parallel=True)
def Z_NCC(left_img, right_img, window_size=9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    print("height: ", height)
    print("width: ", width)
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)

    for left_y in prange(height):
        for left_x in prange(width):
            wleft, wright, wdown, wup = threshold, threshold, threshold, threshold
            if left_x < threshold:
                wleft = left_x
            if left_y < threshold:
                wup = left_y
            if left_x >= width - threshold:
                wright = width - left_x - 1
            if left_y > height - threshold:
                wdown = height - left_y - 1
            window_width = (left_x - wleft, left_x + wright + 1)
            window_height = (left_y - wup, left_y + wdown + 1)
            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]), int(window_width[0]):int(window_width[1])])
            best_match_disparity = 999999999
            score = 0
            for right_x in prange(left_x + 1):
                if right_x - wleft < 0:
                    pass
                elif right_x + wright + 1 > width:
                    pass
                else:
                    right_window_width = (int(right_x - wleft), int(right_x + wright + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1], right_window_width[0]:right_window_width[1]])
                    n = (window_width[1] - window_width[0]) * (window_height[1] - window_height[0])
                    left_mean = np.mean(left_window)
                    right_mean = np.mean(right_window)
                    ncc_score = np.sum(np.multiply(left_window - np.mean(left_window), right_window - np.mean(right_window)))
                    ncc_score = ncc_score / (np.sqrt(np.sum((left_window - left_mean)**2)) * np.sqrt(np.sum((right_window - right_mean)**2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx

