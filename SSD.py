import copy

import cv2
import numpy as np
from numba import njit, prange
import tqdm
import time


@njit(parallel=True)
def compute_SSD(left_image, right_image, win_width, win_height):
    # start_time = time.time()
    # Check if two images has the same shape
    if left_image.shape == right_image.shape:
        height, width = left_image.shape
    else:
        raise Exception("Two images have different shape")
    print("height:", height, "width:", width)

    # Depth (disparity) map
    depth = np.zeros((height, width), np.uint8)
    for i, j in np.ndindex(depth.shape):
        # np.fill not supported by numba
        depth[i, j] = left_image[i, j]
    depth = depth.astype(np.uint8)
    # depth = copy.deepcopy(left_image)

    # half of the window size
    half_w = win_width // 2
    half_h = win_height // 2

    for y in prange(height):

        top = half_h
        bottom = half_h
        # Out of bound
        if y < half_h:
            top = y
        if y + half_h > height:
            bottom = height - y - 1
        h_range = (y - top, y + bottom + 1)

        for l_x in prange(width):

            left = half_w
            right = half_w
            # Out of bound
            if l_x < half_w:
                left = l_x
            if l_x + half_w >= width:
                right = width - l_x - 1

            w_range = (l_x - left, l_x + right + 1)
            left_win = np.ascontiguousarray(left_image[
                                            h_range[0]:h_range[1],
                                            w_range[0]:w_range[1]])
            min = 999999
            disparity = 999999
            for r_x in prange(l_x + 1):

                if r_x < left:
                    pass
                elif r_x + right + 1 > width:
                    pass
                else:
                    r_w_range = (r_x - left, r_x + right + 1)

                    right_win = np.ascontiguousarray(right_image[
                                                     h_range[0]:h_range[1],
                                                     r_w_range[0]:r_w_range[1]])

                    dif = np.sum((left_win - right_win) ** 2)

                    if dif < min:
                        min = dif
                        disparity = np.abs(l_x - r_x)

            depth[y, l_x] = disparity
    # print("--- %s seconds ---" % (time.time() - start_time))
    return depth
def mask_Gaussian(left_image, right_image, win_width, win_height):
    # Check if two images has the same shape
    if left_image.shape == right_image.shape:
        height, width = left_image.shape
    else:
        raise Exception("Two images have different shape")
    print("height:", height, "width:", width)

    # Depth (disparity) map
    depth = np.zeros((height, width), np.uint8)
    for i, j in np.ndindex(depth.shape):
        # np.fill not supported by numba
        depth[i, j] = left_image[i, j]
    depth = depth.astype(np.uint8)
    # depth = copy.deepcopy(left_image)

    # half of the window size
    half_w = win_width // 2
    half_h = win_height // 2

    for y in prange(height):

        top = half_h
        bottom = half_h
        # Out of bound
        if y < half_h:
            top = y
        if y + half_h > height:
            bottom = height - y - 1
        h_range = (y - top, y + bottom + 1)

        for l_x in prange(width):

            left = half_w
            right = half_w
            # Out of bound
            if l_x < half_w:
                left = l_x
            if l_x + half_w >= width:
                right = width - l_x - 1

            w_range = (l_x - left, l_x + right + 1)
            left_win = np.ascontiguousarray(left_image[
                                            h_range[0]:h_range[1],
                                            w_range[0]:w_range[1]])
            mask = cv2.getGaussianKernel(left_win.shape[0], left_win.shape[1])
            left_win = left_win * mask

            min = 999999
            disparity = 999999
            for r_x in prange(l_x + 1):

                if r_x < left:
                    pass
                elif r_x + right + 1 > width:
                    pass
                else:
                    r_w_range = (r_x - left, r_x + right + 1)

                    right_win = np.ascontiguousarray(right_image[
                                                     h_range[0]:h_range[1],
                                                     r_w_range[0]:r_w_range[1]])
                    dif = np.sum((left_win - right_win) ** 2)

                    if dif < min:
                        min = dif
                        disparity = np.abs(l_x - r_x)

            depth[y, l_x] = disparity
    return depth

@njit(parallel=True)
def compute_ZSSD(left_image, right_image, win_width, win_height):
    # start_time = time.time()
    # Check if two images has the same shape
    if left_image.shape == right_image.shape:
        height, width = left_image.shape
    else:
        raise Exception("Two images have different shape")
    print("height:", height, "width:", width)

    # Depth (disparity) map
    depth = np.zeros((height, width), np.uint8)
    for i, j in np.ndindex(depth.shape):
        # np.fill not supported by numba
        depth[i, j] = left_image[i, j]
    depth = depth.astype(np.uint8)
    # depth = copy.deepcopy(left_image)

    # half of the window size
    half_w = win_width // 2
    half_h = win_height // 2

    for y in prange(height):

        top = half_h
        bottom = half_h
        # Out of bound
        if y < half_h:
            top = y
        if y + half_h > height:
            bottom = height - y - 1
        h_range = (y - top, y + bottom + 1)

        for l_x in prange(width):

            left = half_w
            right = half_w
            # Out of bound
            if l_x < half_w:
                left = l_x
            if l_x + half_w >= width:
                right = width - l_x - 1

            w_range = (l_x - left, l_x + right + 1)
            left_win = np.ascontiguousarray(left_image[
                                            h_range[0]:h_range[1],
                                            w_range[0]:w_range[1]])
            min = 999999
            disparity = 999999
            for r_x in prange(l_x + 1):

                if r_x < left:
                    pass
                elif r_x + right + 1 > width:
                    pass
                else:
                    r_w_range = (r_x - left, r_x + right + 1)

                    right_win = np.ascontiguousarray(right_image[
                                                     h_range[0]:h_range[1],
                                                     r_w_range[0]:r_w_range[1]])

                    l = left_win-(np.mean(left_win))
                    r = right_win-(np.mean(right_win))
                    dif = np.sum((l-r) ** 2)

                    if dif < min:
                        min = dif
                        disparity = np.abs(l_x - r_x)

            depth[y, l_x] = disparity
    # print("--- %s seconds ---" % (time.time() - start_time))
    return depth