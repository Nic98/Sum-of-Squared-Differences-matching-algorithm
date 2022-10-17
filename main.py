# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numba import njit, prange
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    truth_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-disparity.png', -1) / 256
    left_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-left.jpg')
    right_image = cv2.imread('./Dataset/2018-07-09-16-11-56_2018-07-09-16-11-56-702-right.jpg')
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY).astype(np.int64)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY).astype(np.int64)

    plt.subplot(2, 1, 1)
    plt.imshow(left_image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(right_image, cmap='gray')
    plt.axis('off')
    plt.show()
    pass

