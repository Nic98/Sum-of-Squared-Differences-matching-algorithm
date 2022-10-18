import copy
import numpy as np

def compute_SSD(left_image, right_image, win_width, win_height):

    # Check if two images has the same shape
    if left_image.shape == right_image.shape:
        height, width = left_image.shape
    else:
        raise Exception("Two images have different shape")
    print("height:", height, "width:", width)

    # Depth (disparity) map
    # depth = np.zeros((height, width), np.uint8)
    depth = copy.deepcopy(left_image)
    # half of the window size
    half_w = win_width // 2
    half_h = win_height // 2

    for y in range(height):

        top = half_h
        bottom = half_h
        # Out of bound
        if y < half_h:
            top = y
        if y + half_h > height:
            bottom = height - y - 1
        h_range = (y - top, y + bottom + 1)
        for l_x in range(width):

            left = half_w
            right = half_w
            # Out of bound
            if l_x < half_w:
                left = l_x
            if l_x + half_w > width:
                right = width - l_x - 1

            w_range = (l_x-left, l_x+right+1)
            left_win = np.ascontiguousarray(left_image[
                                            h_range[0]:h_range[1],
                                            w_range[0]:w_range[1]])
            min = 999999
            disparity = 999999
            for r_x in range(l_x+1):

                if r_x < left:
                    pass
                elif r_x + right + 1 > width:
                    pass
                else:
                    r_w_range = (r_x-left, r_x+right+1)

                    right_win = np.ascontiguousarray(right_image[
                                                     h_range[0]:h_range[1],
                                                     r_w_range[0]:r_w_range[1]])

                    dif = np.sum((left_win-right_win)**2)

                    if dif < min:
                        min = dif
                        disparity = np.abs(l_x - r_x)

            depth[y, l_x] = disparity

        return depth
