import numpy as np

def compute_SSD(left_image, right_image, win_size):

    # Check if two images has the same shape
    if left_image.shape == right_image.shape:
        width, height = left_image.shape
    else:
        raise Exception("Two images have different shape")

    # Depth (disparity) map
    depth = np.zeros((width, height), np.unit8)
    depth.shape = height, width

    # Half of the window size
    half = int(win_size/2)

    for y in range(height):
        for x in range(width):
                # Construct a window



    return