from evaluation import *
from NCC import *
from matplotlib import pyplot as plt
def evaluate_diff_size(left_image, right_image, truth_image):
    rms_l = []
    xaxis = range(10, 120, 10)

    for x in xaxis:
        disparity = Z_NCC(left_image, right_image, x)
        rms_l.append(evaluate(disparity, truth_image))

    plt.plot(xaxis, rms_l)
    plt.plot(xaxis, rms_l)
    plt.show()
    return
