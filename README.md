# COMP90086Project
## SSD.py
This file is performing a Sum of Squared Differences matching algorithm.

Function **ssd** is the basic sum of squared difference algorithm implementation. It requires 5 parameters to run the function:
1. Left image
2. Right image
3. Matching window width
4. Matching window height
5. SSD or Zero Mean of SSD

The function will output a depth(disparity) map.

Function **mask_Gaussian** is built on the ssd function. However, it uses a guassian kernel is mask the left and right window before matching.
It requires 4 parameters to run the function:
1. Left image
2. Right image
3. Matching window width
4. Matching window height

The function will output a depth(disparity) map with both left and right be masked with a gaussian kernel.

Function **compute_ZSSD_smooth** is implemented base on the idea of smoothing. When comparing two windows, 
an extra parameter 'smoothness' will be added for smoothing purpose.
It requires 6 parameters to run the function:
1. Left image
2. Right image
3. A disparity graph of two given images
4. Weight of the smoothness
5. Matching window width
6. Matching window height

The function will output a depth(disparity) map with smoothing.

## winSize.py
This file is to generate a plot for observing the relationship of the window size and root mean squared error.

Function **evaluate_diff_size** requires 3 parameters:
1. Left image
2. Right image
3. Ground Truth image

The function will plot a line chart illustrates the relationship between window size and rms.
