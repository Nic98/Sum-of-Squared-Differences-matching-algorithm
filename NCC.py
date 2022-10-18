import numpy as np
from math import floor
from copy import deepcopy
from numba import njit, prange


@njit(parallel=True)
def Original_NCC(left_img, right_img, window_size=9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_dict = dict()
    disparity_mtx = np.zeros((width, height))
    threshold = floor(window_size / 2)
    for left_y in height:
        for left_x in width:
            wleft, wright, wdown, wup = threshold
            if left_x < threshold:
                wleft = left_x
            if left_y < threshold:
                wup = left_y
            if left_x > width - threshold:
                wright = width - left_x - 1
            if left_y > height - threshold:
                wdown = height - left_y - 1
            window_width = (left_x - wleft, left_x + wright + 1)
            window_height = (left_y - wup, left_y + wdown + 1)
            left_window = np.ascontiguousarray(
                left_img[window_width[0]:window_width[1], window_height[0]:window_height[1]])
            max_score = 99999
            score = 0

            for right_x in range(0, left_x + 1):
                if right_x > left_y and right_x + wright + 1 < width:
                    right_window = np.ascontiguousarray(
                        (right_img[window_width[0]:window_width[1], window_height[0]:window_height[1]]))
                    ncc_score = 1 / (window_width[0] * window_height[0]) * np.multiply(left_window,
                                                                                       right_window) * 1 / (
                                        np.std(left_window) * np.std(right_window))
                    if ncc_score > max_score:
                        max_score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)

            disparity_mtx[left_x, left_y] = best_match_disparity
    return disparity_mtx
