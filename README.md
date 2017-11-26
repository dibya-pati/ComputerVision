# Computer Vision
Segmentation,Object Detection and many more

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
    1. It first uses SLIC(Simple Linear Iterative Clustering) to form superpixels, the segmentation(n) and compactness parameter of SLIC can be adjusted to get the desired segment size and shape.
    2. Read the image markings from the file where blue marks the foreround and red the background, these pixel markings are used to creat the source and sink nodes for the graph-cut to separate the image into two classes (the foreground and background).
    3. Use Energy based image segmentation by comparing the image histograms with the intial set of background and foreground histograms(from markings) 
    4. Perform normalized graph cut to the pixels to background(bg) and foreground(fg)
    5. In the extra section the markings are read from a gui, where user can interactively draws the bg and fg markings, use 'c' or 'C' to toggle between color of marking and once done press 'ESC' to compute the segmented mask
 5. HW5:
In this section we experimented with 3-d reconstruction using structured light. The test comprises of projecting a series of harizontal and vertical strips of light on the object & then reading the pattern using a standard camera. Here, we consider a series of images for 3-d recontruction
    1. Read the images and populate the scan value for each pixel(read a set a horizonal then vertical strips that form LSB and MSB of the binary code)
    2. Use the lookup table to find the (x,y) position from the projector
    3. Compute the undistorted values of (x,y) using the given K-matrix and distortion matrix of the projector and camera respectively
    4. We have two sets (x,y) for a given pixel One from the projector and one from the camera's peprspective	
    5. Use OpenCV triangulation to determine 3-d coordinate(homogenous space) for the a given (x,y)
    6. Compute the actual 3-d point from the homogenous matrix using OpenCV convert from homogenous api
    7. Result 3-d location for a 2-d point