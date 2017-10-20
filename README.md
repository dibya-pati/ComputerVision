# Computer Vision

This repo comprises of Python Open CV implementation of :

1.Histogram equalization/Contrast Enhancement:This is based on Prince's approach of linearizing the channels by multiplying
it with color depth i.e. 255 for 8bit, and dividing it with the number of pixels, more precisely eq 13.3 and 13.4 from 13.1.2 Histogram equalization of the book Computer vision

<img src="https://latex.codecogs.com/gif.latex?c_{k}=\frac{\sum_{l=1}^{k}h_{l}}{IJ}" title="c_{k}=\frac{\sum_{l=1}^{k}h_{l}}{IJ}" />

where I and J are number of pixels in x and y respectively
and then maping each pixel to its percentile value by,

<img src="https://latex.codecogs.com/gif.latex?x_{ij}=Kc_{p_{ij}}" title="x_{ij}=Kc_{p_{ij}}" />

where K is the maximum intensity(if 8bit its 255 and so on)


2.Filter: Implemetation of Low pass and High Pass filter using Fast Fourier Transform(FFT), Convolution and again Inverse FFT

3.Deblurring : this involves using deconvolution i.e. dividing of image with the original gaussian kernel used for blurring

4.Laplacian Pyramid blending: Perform blending of two images by creating pyramids and adding the LP and HP components at each level and traversing up the pyramid

HW2:
This exercise involves image stitching after having done affine(after converting to cylindrical coordinates) and homography transformations.It uses Lunchroom image : PASSTA Dataset

HW3:
Object tracking using:
1.Haar cascades:Voila-Jones detectors
2.Camshift detector(modified mean shift)
3.Haar features with Kalman corrections
4.particle filters(from Lucas and Kanade)
5.Optical Flow(using Shi and Tomasi )
