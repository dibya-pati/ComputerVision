# Computer Vision
Open CV implementations of segmentation,Pyramid Blending,Object Detection and many more

This repo comprises of Python Open CV implementation of :
 1. HW1:

    1. Histogram equalization/Contrast Enhancement:This is based on Prince's approach of linearizing the channels by multiplying
it with color depth i.e. 255 for 8bit, and dividing it with the number of pixels, more precisely eq 13.3 and 13.4 from 13.1.2 Histogram equalization of the book Computer vision <img src="https://latex.codecogs.com/gif.latex?c_{k}=\frac{\sum_{l=1}^{k}h_{l}}{IJ}" title="c_{k}=\frac{\sum_{l=1}^{k}h_{l}}{IJ}" /> where I and J are number of pixels in x and y respectively and then maping each pixel to its percentile value by, <img src="https://latex.codecogs.com/gif.latex?x_{ij}=Kc_{p_{ij}}" title="x_{ij}=Kc_{p_{ij}}" /> where K is the maximum intensity(if 8bit its 255 and so on)
    2. Filter: Implemetation of Low pass and High Pass filter using Fast Fourier Transform(FFT), Convolution and again Inverse FFT
    3. Deblurring and dividing gaussian : this involves using deconvolution i.e. dividing of image with the original gaussian kernel used for blurring.
    4. Laplacian Pyramid blending: Perform blending of two images by creating pyramids and adding the LP and HP components at each level and traversing up the pyramid.
 2. HW2:
This exercise involves image stitching after having done affine(after converting to cylindrical coordinates) and homography transformations.It uses Lunchroom image : PASSTA Dataset
 3. HW3:
This execrcise implements object tracking and optical flow in OpenCV.
    1. Haar cascades:Voila-Jones detectors
    2. Camshift detector(modified mean shift)
    3. Haar features with Kalman corrections
    4. particle filters(from Lucas and Kanade)
    5. Optical Flow(using Shi and Tomasi )
 4. HW4:
This execrcise primarily concentrates on semi-supervised image segmentation on astronaut image.
It first uses SLIC(Simple Linear Iterative Clustering) to form superpixels, the segmentation(n) and compactness parameter of SLIC can be adjusted to get the desired segment size and shape.After that the system reads the image markings from the file where blue marks the foreround and red the background, these pixel markings are used to created the source and sink nodes for the graph-cut to separate the image into two classes (the foreground and background).In this exercise we have used Energy based image segmentation by comparing the image histograms with the intial set of background and foreground histograms(from markings) followed by normalized graph cut to the pixels to bg and fg.In the bonus section the markings are read from a gui, where user can interactively draw the bg and fg markings, use 'c' or 'C' to togle between color of marking and once done press 'ESC' to compute the segmented mask
