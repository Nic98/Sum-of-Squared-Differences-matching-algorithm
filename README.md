# COMP90086Project
### Note
1. Because it is very time consuming to run all of our functions in their entirety, most of the data you see in the report is based on operations with a window of size 20. The results of our operations on a window of size 50 are recorded in Result1.png in the zip archive.
2. The main function is the main file that runs our methods and in it you can see how the functions we have designed will run. In addition to this, you are free to adjust a number of parameters such as the window size, the details of how to do this are documented at the bottom of the README.
3. As mentioned above, since the program runs too long, we predict the run time by our algorithm on each image by using the tqdm module. In addition, we also apply it to check the performance of the algorithm after we accelerated the process.
4. There are a few pictrues in the zip file, they are the screenshot of our result. Since we are not using Jupyternotebook to wirte our code this time, please use these images to verify our program.
### Functions
## main.py
**main.py** is the main file to run our program. You can find how we import and transform the input image. Besides, all the functions that are used to display the final result are put into this file. You can simply run the program and get the same result as what is shown in the Result1.png and Disparity1.png.
## smoothnessEval.py
smoothnessEval.py is used to compare the impact of window size on the performance of our algorithm. The corresponding output is recorded in smoothnessEval.png
## evaluation.py
evaluation.py is where we write our evaluate function.

**evaluate(disparity, truth)** takes the predicted disparity array and the ground truth disparity as input, it will return a 6 elements tuple: (error 0.25, error 0.5, , error 1, error 2, error 4, rms)

## NCC.py
NCC.py contains the main implementations of NCC match algorithm

**NCC(left_img, right_img, windowsize=9)** returns the disparity using Normalized cross-correlation algorithm.

**ZNCC(left_img, right_img, windowsize=9)** returns the disparity using Zero Means Normalized cross-correlation algorithm.

**filter_match(left_img, right_img, windowsize=10, algo='ncc')** returns the disparity using NCC/ZNCC and the weighted kernel we designed.

**disparity_smooth(left_img, right_img, disparity, windowsize=10, algo='ncc', weight=0.9)** returns the disparity using NCC/ZNCC and smoothing, the last argument is used to set the weight factor lambda.

#### SAD.py
SAD.py contains the main implementations of NCC match algorithm

**compute_SAD(left_img, right_img, win_width, win_height)** returns the disparity using basic SAD algorithm.

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