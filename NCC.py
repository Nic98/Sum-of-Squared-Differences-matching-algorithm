import numpy as np
from math import floor
from copy import deepcopy
from numba import njit, prange
import tqdm

# @njit(parallel=True)
def NCC(left_img, right_img, window_size=9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    x_range = 120
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)

    for left_y in tqdm.tqdm(range(height)):
        for left_x in prange(width):
            left_bound, right_bound, bottom_bound, top_bound = threshold, threshold, threshold, threshold
            if left_x < threshold:
                left_bound = left_x
            if left_y < threshold:
                top_bound = left_y
            if left_x >= width - threshold:
                right_bound = width - left_x - 1
            if left_y > height - threshold:
                bottom_bound = height - left_y - 1
            left_window_width = (left_x - left_bound, left_x + right_bound + 1)
            window_height = (left_y - top_bound, left_y + bottom_bound + 1)
            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]),
                int(left_window_width[0]):int(left_window_width[1])])
            best_match_disparity = 9999
            score = 0
            for right_x in prange(left_x - x_range, left_x + 1):
                if right_x - left_bound < 0:
                    pass
                elif right_x + right_bound + 1 > width:
                    pass
                else:
                    right_left_window_width = (int(right_x - left_bound), int(right_x + right_bound + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1],
                        right_left_window_width[0]:right_left_window_width[1]])
                    n = (left_window_width[1] - left_window_width[0]) * (window_height[1] - window_height[0])
                    ncc_score = np.sum(np.multiply(left_window, right_window))
                    ncc_score = ncc_score / (np.sqrt(np.sum(left_window ** 2)) * np.sqrt(np.sum(right_window ** 2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx


# @njit(parallel=True)
def ZNCC(left_img, right_img, window_size=9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    x_range = 120
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)
    left_mean = np.mean(left_img)
    right_mean = np.mean(right_img)
    for left_y in tqdm.tqdm(range(height)):
        for left_x in range(width):
            left_bound, right_bound, bottom_bound, top_bound = threshold, threshold, threshold, threshold
            if left_x < threshold:
                left_bound = left_x
            if left_y < threshold:
                top_bound = left_y
            if left_x >= width - threshold:
                right_bound = width - left_x - 1
            if left_y > height - threshold:
                bottom_bound = height - left_y - 1
            left_window_width = (left_x - left_bound, left_x + right_bound + 1)
            window_height = (left_y - top_bound, left_y + bottom_bound + 1)
            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]),
                int(left_window_width[0]):int(left_window_width[1])])
            best_match_disparity = 9999
            score = 0
            for right_x in range(left_x - x_range, left_x + 1):
                if right_x - left_bound < 0:
                    pass
                elif right_x + right_bound + 1 > width:
                    pass
                else:
                    right_left_window_width = (int(right_x - left_bound), int(right_x + right_bound + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1],
                        right_left_window_width[0]:right_left_window_width[1]])
                    n = (left_window_width[1] - left_window_width[0]) * (window_height[1] - window_height[0])
                    left_mean = np.mean(left_window)
                    right_mean = np.mean(right_window)
                    ncc_score = np.sum(np.multiply(left_window - left_mean, right_window - right_mean))
                    left_var = np.std(left_window)  # (np.sqrt(left_window**2 - left_mean **2))
                    right_var = np.std(right_window)  # (np.sqrt(right_window ** 2 - right_mean ** 2))
                    const = 1/((2 * left_window.shape[0] * left_window[1] + 1)**2)
                    ncc_score = np.sum(ncc_score / (left_var * right_var * const))
                    # ncc_score = ncc_score / (np.sqrt(np.sum((left_window - left_mean) ** 2)) * np.sqrt(np.sum((right_window - right_mean) ** 2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx


@njit(parallel=True)
def filter_match(left_img, right_img, window_size=10, algo='ncc'):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    x_range = 120
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)
    for left_y in prange(height):
        for left_x in prange(width):
            left_bound, right_bound, bottom_bound, top_bound = threshold, threshold, threshold, threshold
            if left_x < threshold:
                left_bound = left_x
            if left_y < threshold:
                top_bound = left_y
            if left_x >= width - threshold:
                right_bound = width - left_x - 1
            if left_y > height - threshold:
                bottom_bound = height - left_y - 1
            left_window_width = (left_x - left_bound, left_x + right_bound + 1)
            window_height = (left_y - top_bound, left_y + bottom_bound + 1)

            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]),
                int(left_window_width[0]):int(left_window_width[1])])
            w_shape = left_window.shape
            my_filter = np.zeros(w_shape, float)
            y_axis = w_shape[1]
            x_axis = w_shape[0]
            for pt1 in prange(y_axis):
                for pt2 in prange(x_axis):
                    dist = np.sqrt(pt1 ** 2 + pt2 ** 2) - 1
                    discount = 1 - ((1 / max(y_axis, x_axis)) * dist)  # * 0.3
                    if discount < 0:
                        discount = 0
                    my_filter[pt2, pt1] = discount
            for i in prange(y_axis):
                if i != 0:
                    my_filter[0, i] = 1
                else:
                    my_filter[0, i] = 1
            my_filter[0, 0] = 1
            if not (w_shape[0] == window_size and w_shape[0] == window_size):
                my_filter = np.ones(w_shape, float)
            left_window = left_window * my_filter
            best_match_disparity = 9999
            score = 0
            for right_x in prange(left_x - x_range, left_x + 1):
                if right_x - left_bound < 0:
                    pass
                elif right_x + right_bound + 1 > width:
                    pass
                else:
                    right_left_window_width = (int(right_x - left_bound), int(right_x + right_bound + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1],
                        right_left_window_width[0]:right_left_window_width[1]])
                    right_window = right_window * my_filter

                    if (algo == 'zncc'):
                        n = (left_window_width[1] - left_window_width[0]) * (window_height[1] - window_height[0])
                        left_mean = np.mean(left_window)
                        right_mean = np.mean(right_window)
                        ncc_score = np.sum(np.multiply(left_window - left_mean, right_window - right_mean))
                        left_var = np.std(left_window)  # (np.sqrt(left_window**2 - left_mean **2))
                        right_var = np.std(right_window)  # (np.sqrt(right_window ** 2 - right_mean ** 2))
                        const = 1 / ((2 * left_window.shape[0] * left_window[1] + 1) ** 2)
                        ncc_score = np.sum(ncc_score / (left_var * right_var * const))
                    else:
                        ncc_score = np.sum(np.multiply(left_window, right_window))
                        ncc_score = ncc_score / (np.sqrt(np.sum(left_window ** 2)) * np.sqrt(np.sum(right_window ** 2)))
                    # ncc_score = ncc_score / (np.sqrt(np.sum((left_window - left_mean) ** 2)) * np.sqrt(np.sum((right_window - right_mean) ** 2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx



@njit(parallel=True)
def disparity_smooth(left_img, right_img, disparity, window_size=10, algo='ncc', weight = 0.9):
    width = left_img.shape[1]
    height = left_img.shape[0]
    disparity_mtx = np.zeros((height, width))
    threshold = int(floor(window_size / 2))
    x_range = 120
    for i, j in np.ndindex(disparity_mtx.shape):
        # np.fill not supported by numba
        disparity_mtx[i, j] = left_img[i, j]
    disparity_mtx = disparity_mtx.astype(np.uint8)
    for left_y in prange(height):
        for left_x in prange(width):
            left_bound, right_bound, bottom_bound, top_bound = threshold, threshold, threshold, threshold
            if left_x < threshold:
                left_bound = left_x
            if left_y < threshold:
                top_bound = left_y
            if left_x >= width - threshold:
                right_bound = width - left_x - 1
            if left_y > height - threshold:
                bottom_bound = height - left_y - 1
            left_window_width = (left_x - left_bound, left_x + right_bound + 1)
            window_height = (left_y - top_bound, left_y + bottom_bound + 1)

            left_window = np.ascontiguousarray(
                left_img[int(window_height[0]):int(window_height[1]),
                int(left_window_width[0]):int(left_window_width[1])])
            w_shape = left_window.shape
            my_filter = np.zeros(w_shape, float)
            y_axis = w_shape[1]
            x_axis = w_shape[0]
            for pt1 in prange(y_axis):
                for pt2 in prange(x_axis):
                    dist = np.sqrt(pt1 ** 2 + pt2 ** 2) - 1
                    discount = 1 - ((1 / max(y_axis, x_axis)) * dist)  # * 0.3
                    my_filter[pt2, pt1] = discount
            for i in prange(y_axis):
                if i != 0:
                    my_filter[0, i] = 1
                else:
                    my_filter[0, i] = 1
            my_filter[0, 0] = 1
            if not (w_shape[0] == window_size and w_shape[0] == window_size):
                my_filter = np.ones(w_shape, float)
            # print(my_filter)
            #left_window = left_window * my_filter
            # print(filtered_window)
            best_match_disparity = 9999
            score = 0
            for right_x in prange(left_x - x_range, left_x + 1):
                if right_x - left_bound < 0:
                    pass
                elif right_x + right_bound + 1 > width:
                    pass
                else:
                    right_left_window_width = (int(right_x - left_bound), int(right_x + right_bound + 1))
                    right_window = np.ascontiguousarray(
                        right_img[window_height[0]:window_height[1],
                        right_left_window_width[0]:right_left_window_width[1]])

                    diff = np.abs(left_x - right_x)
                    smoothness = weight * (np.abs(diff - disparity[left_y - 1, left_x]) +
                                           np.abs(diff - disparity[left_y + 1, left_x]) +
                                           np.abs(diff - disparity[left_y, left_x + 1]) +
                                           np.abs(diff - disparity[left_y, left_x - 1]) +
                                           np.abs(diff - disparity[left_y - 1, left_x - 1]) +
                                           np.abs(diff - disparity[left_y - 1, left_x + 1]) +
                                           np.abs(diff - disparity[left_y + 1, left_x - 1]) +
                                           np.abs(diff - disparity[left_y + 1, left_x + 1]))

                    #right_window = right_window * my_filter

                    if (algo == 'zncc'):
                        n = (left_window_width[1] - left_window_width[0]) * (window_height[1] - window_height[0])
                        left_mean = np.mean(left_window)
                        right_mean = np.mean(right_window)
                        ncc_score = np.multiply(left_window - left_mean, right_window - right_mean)
                        left_var = np.std(left_window)  # (np.sqrt(left_window**2 - left_mean **2))
                        # print(left_var)
                        right_var = np.std(right_window)  # (np.sqrt(right_window ** 2 - right_mean ** 2))
                        # print(ncc_score)
                        # print((left_var * right_var + 1))
                        # print(right_var)
                        ncc_score = np.sum(ncc_score / (left_var * right_var)) - smoothness
                    else:
                        ncc_score = np.sum(np.multiply(left_window, right_window))
                        ncc_score = ncc_score / (np.sqrt(np.sum(left_window ** 2)) * np.sqrt(np.sum(right_window ** 2))) - smoothness
                    # ncc_score = ncc_score / (np.sqrt(np.sum((left_window - left_mean) ** 2)) * np.sqrt(np.sum((right_window - right_mean) ** 2)))
                    if ncc_score > score:
                        score = ncc_score
                        best_match_disparity = np.abs(left_x - right_x)
            disparity_mtx[left_y, left_x] = best_match_disparity
    return disparity_mtx