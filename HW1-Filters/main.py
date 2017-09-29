# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
import math

# import copy
# from matplotlib import pyplot as plt


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    img_out = img_in  # Histogram equalization result
    try:
        b, g, r = cv2.split(img_in)

        # using range instead of single number i.e. 256 to indicate a bin width of unity from 0 through 256
        # This creates 255 different bins, list(range(257)) is equivalent of saying 256
        hist_b = np.histogram(b, list(range(257)))[0]
        hist_g = np.histogram(g, list(range(257)))[0]
        hist_r = np.histogram(r, list(range(257)))[0]

        cdf_b = np.cumsum(hist_b)
        cdf_g = np.cumsum(hist_g)
        cdf_r = np.cumsum(hist_r)

        cdf_b_normalized = cdf_b * 255 / (cdf_b.max())
        cdf_g_normalized = cdf_g * 255 / (cdf_g.max())
        cdf_r_normalized = cdf_r * 255 / (cdf_r.max())

        img_out = cv2.merge((cdf_b_normalized[b], cdf_g_normalized[g],
                             cdf_r_normalized[r]))

    except:
        print('grayscale')
    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def high_pass_filter(img_in):

    # Write high pass filter here
    img_out = img_in  # Low pass filter result
    filter_size = 20
    try:
        b, g, r = cv2.split(img_in)
        #convert image to fft
        fshift_b = ft(b)
        fshift_g = ft(g)
        fshift_r = ft(r)

        rows, cols, channels = img_in.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        fshift_b[crow - filter_size:crow + filter_size, ccol - filter_size:
                 ccol + filter_size] = 0
        fshift_g[crow - filter_size:crow + filter_size, ccol - filter_size:
                 ccol + filter_size] = 0
        fshift_r[crow - filter_size:crow + filter_size, ccol - filter_size:
                 ccol + filter_size] = 0

        #bring back the DC components to top left corner
        img_back = cv2.merge((np.abs(ift(fshift_b)),np.abs(ift(fshift_g)),\
                   np.abs(ift(fshift_r))))

        img_out = img_back

    except:

        #would still fail, failing to read grayscale, so, nothing reaches to this point
        fshift_b = ft(img_in)

        rows, cols = img_in.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift_b[crow - filter_size:crow + filter_size, ccol - filter_size:
                 ccol + filter_size] = 0

        img_back = (np.abs(ift(fshift_b)))
        img_out = img_back

    return True, img_out


def low_pass_filter(img_in):
    # Write low pass filter here
    img_out = img_in  # High pass filter result
    filter_size = 20
    try:
        b, g, r = cv2.split(img_in)
        # convert image to fft
        fshift_b = ft(b)
        fshift_g = ft(g)
        fshift_r = ft(r)

        rows, cols, channels = img_in.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        fshift_mask = np.zeros(shape=(rows, cols), dtype=int)
        fshift_mask[crow - filter_size:crow + filter_size, ccol - filter_size:
                    ccol + filter_size] = 1

        fshift_b = np.multiply(fshift_mask, fshift_b)
        fshift_g = np.multiply(fshift_mask, fshift_g)
        fshift_r = np.multiply(fshift_mask, fshift_r)

        img_back = cv2.merge((np.abs(ift(fshift_b)), np.abs(ift(fshift_g)),
                              np.abs(ift(fshift_r))))

        img_out = img_back

    except:
        # would still fail, failing to read grayscale, so, nothing reaches to this point
        fshift_b = ft(img_in)
        rows, cols = img_in.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        fshift_mask = np.zeros(shape=(rows, cols), dtype=int)
        fshift_mask[crow - filter_size:crow + filter_size, ccol - filter_size:
                    ccol + filter_size] = 1

        fshift_b = np.multiply(fshift_mask, fshift_b)
        img_back = (np.abs(ift(fshift_b)))

        img_out = img_back
    return True, img_out


#to complete deconvolution from the convoluted image
def deconvolution(img_in):

    # Write deconvolution codes here
    img_out = img_in
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    gkf = ft(gk, (img_in.shape[0],
                  img_in.shape[1]))  # so we can multiple easily

    try:
        b, g, r = cv2.split(img_in)

        imf_b = ft(b, (img_in.shape[0],
                       img_in.shape[1]))  # make sure sizes match
        imf_g = ft(g, (img_in.shape[0],
                       img_in.shape[1]))  # make sure sizes match
        imf_r = ft(r, (img_in.shape[0],
                       img_in.shape[1]))  # make sure sizes match

        imconvf_b = np.divide(imf_b, gkf)
        imconvf_g = np.divide(imf_g, gkf)
        imconvf_r = np.divide(imf_r, gkf)

        # now for example we can reconstruct the corrected image from its FT
        corrected_b = ift(imconvf_b)
        corrected_g = ift(imconvf_g)
        corrected_r = ift(imconvf_r)

        img_out = cv2.merge((np.abs(corrected_b), np.abs(corrected_g),
                             np.abs(corrected_r)))

    except:

        imf_b = ft(img_in, (img_in.shape[0],
                            img_in.shape[1]))  # make sure sizes match

        imconvf_b = np.divide(imf_b, gkf)

        # now for example we can reconstruct the blurred image from its FT
        corrected_b = ift(imconvf_b)

        img_out = np.abs(corrected_b)

    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.png"

    output_image3 = output_image3[:] * 255 / (
        output_image3.max() - output_image3.min())

    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here
    img_out = img_in1  # Blending result

    # generate Gaussian pyramid for A & B
    GA = img_in1.copy()
    GB = img_in2.copy()
    gpA = [GA]
    gpB = [GB]

    for i in range(6):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        gpA.append(GA)
        gpB.append(GB)

    # generate Laplacian Pyramid for A

    lpA = [gpA[len(gpA) - 1]]
    lpB = [gpB[len(gpA) - 1]]

    for i in range(len(gpA) - 1, 0, -1):
        GE_A = cv2.pyrUp(
            gpA[i], dstsize=(gpA[i - 1].shape[0], gpA[i - 1].shape[1]))
        GE_B = cv2.pyrUp(
            gpB[i], dstsize=(gpB[i - 1].shape[0], gpB[i - 1].shape[1]))
        L_A = cv2.subtract(gpA[i - 1], GE_A)
        L_B = cv2.subtract(gpB[i - 1], GE_B)

        lpA.append(L_A)
        lpB.append(L_B)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_, dstsize=(LS[i].shape[0], LS[i].shape[1]))
        ls_ = cv2.add(ls_, LS[i])

    return True, ls_


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
    input_image1 = input_image1[:, :input_image1.shape[0]]
    input_image2 = input_image2[:input_image1.shape[0], :input_image1.shape[0]]
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
