import numpy as np

def evaluate(disparity, truth):
    valid = truth > 0
    val = np.sum(valid)
    disparity *= valid
    inval = np.sum(truth == 0)

    error = np.abs(disparity - truth)
    error_0_25 = cal_error(error, 0.25, inval, val)
    error_0_5 = cal_error(error, 0.5, inval, val)
    error_1 = cal_error(error, 1, inval, val)
    error_2 = cal_error(error, 2, inval, val)
    error_4 = cal_error(error, 4, inval, val)

    print(error_0_25, error_0_5, error_1, error_2, error_4)
    rms = np.sqrt(np.sum((disparity - truth) ** 2) / np.sum(valid))
    print(rms)

def cal_error(error, pixel, inval, val):
    res = np.sum(error < pixel) - inval / val
    return res