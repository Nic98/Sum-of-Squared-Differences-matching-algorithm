import numpy as np

def evaluate(img_disparity, img_gt):
    valid = img_gt > 0
    valid_num = np.sum(valid)

    img_disparity = img_disparity * valid

    invalid_num = np.sum(img_gt == 0)

    error = np.abs(img_disparity - img_gt)
    error_0_25 = (np.sum(error < 0.25) - invalid_num) / valid_num
    error_0_5 = (np.sum(error < 0.5) - invalid_num) / valid_num
    error_1 = (np.sum(error < 1) - invalid_num) / valid_num
    error_2 = (np.sum(error < 2) - invalid_num) / valid_num
    error_4 = (np.sum(error < 4) - invalid_num) / valid_num
    print(error_0_25, error_0_5, error_1, error_2, error_4)

    rms = np.sqrt(np.sum((img_disparity - img_gt) ** 2) / np.sum(valid))
    print(rms)