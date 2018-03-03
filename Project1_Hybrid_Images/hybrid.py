import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math 

def singleImage_helper(img, kernel, is_convolution):
    (row, col) = img.shape
    (k, l) = kernel.shape
    k_center = (int)(k / 2)
    l_center = (int)(l / 2)
    res = np.zeros((row,col))
    for i in range (row):
        for j in range (col):
            for m in range (k):
                for n in range (l):
                    if is_convolution:
                        m1 = i - (m - k_center)
                        n1 = j - (n - l_center)
                    else:
                        m1 = i + (m - k_center)
                        n1 = j + (n - l_center)
                    if (m1 < row and m1 >= 0)  and (n1 < col and n1 >= 0):
                        img_f = img[m1][n1] 
                        res[i][j] += kernel[m][n] * img_f

    return res


def rgb_helper(img, kernel, is_convolution):
    (row, col, v ) = img.shape
    res = np.zeros((row,col,v))
    for i in range (v):
        res[:, :, i] = singleImage_helper(img[:, :, i], kernel, is_convolution)
    return res


# res = np.random.rand(30,20,3)
# temp = res[:,:,1]
# print (temp.shape)

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    dimension = len(img.shape)
    if dimension == 2:
        return singleImage_helper(img, kernel, False)
    else:
        return rgb_helper(img, kernel, False)
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    dimension = len(img.shape)
    if dimension == 2:
        return singleImage_helper(img, kernel, True)
    else:
        return  rgb_helper(img, kernel, True)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):

    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    res = np.zeros((width, height))
    w_center = int(width / 2)
    h_center = int(height / 2)
    sigma_square = sigma ** 2

    matrix_sum = 0.0

    for i in range (width):
        for j in range (height):
            dist_square = ((i - w_center) ** 2) + ((j - h_center) ** 2)
            res[i][j] = (1.0 / (2.0 * math.pi * sigma_square)) * (math.e ** (-dist_square / (2.0 * sigma_square))) 
            matrix_sum += res[i][j]
    
    for i in range (width):
        for j in range (height):
            res[i][j] = res[i][j] / matrix_sum
    return res
    
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    gaussian_kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, gaussian_kernel)
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    #-------------------------------wrong -------------------------------
    res = img - low_pass(img, sigma, size)
    return res

    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


