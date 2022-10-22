import numpy as np
from copy import deepcopy
def evaluate(disparity, truth):
    shape = disparity.shape
    remove_zero = deepcopy(truth)
    zero_counter = 0
    true_counter = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if truth[i, j] > 0 :
                true_counter += 1
                remove_zero[i, j] = True
            else:
                zero_counter += 1
                remove_zero[i, j] = False
    disparity = disparity * remove_zero
    error = np.abs(disparity - truth)
    error_list = [0.25, 0.5, 1, 2, 4]
    result_list = []
    for i in error_list:
        result_list.append((np.sum(error < i) - zero_counter)/true_counter)
    error025 = result_list[0]
    error05 = result_list[1]
    error1 = result_list[2]
    error2 = result_list[3]
    error4 = result_list[4]
    rms = np.sqrt(np.sum((disparity - truth) ** 2) / np.sum(true_counter))
    return error025, error05, error1, error2, error4, rms