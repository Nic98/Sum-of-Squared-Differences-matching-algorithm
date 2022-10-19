from evaluation import *
from NCC import *
from matplotlib import pyplot as plt


def evaluate_diff_size(left_image, right_image, truth_image):
    rms_l = []
    xaxis = range(10, 120, 10)

    for x in xaxis:
        print(x)
        zncc_disparity = Z_NCC(left_image, right_image, x)
        rms_l.append(evaluate(zncc_disparity, truth_image))
    # zncc_disparity = Z_NCC(left_image, right_image, 10)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 20)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 30)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 40)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 50)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 60)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 70)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 80)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 90)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 100)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 110)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    #
    # zncc_disparity = Z_NCC(left_image, right_image, 120)
    # rms_l.append(evaluate(zncc_disparity, truth_image))
    plt.plot(xaxis, rms_l)
    plt.plot(xaxis, rms_l)
    plt.show()
    return
