import numpy as np
import cv2

# ge the amplitude of the filtered image
def get_amplitude(filtered_img):
	length = len(filtered_img) * len(filtered_img[0])
	amplitude = 0
	for item in filtered_img:
		for elem in item:
			amplitude+=abs(elem)
	return amplitude/length

# return the list of gabor features
def gabor_filter(img):
	# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
	# ksize - size of gabor filter (n, n)
	# sigma - standard deviation of the gaussian function
	# theta - orientation of the normal to the parallel stripes
	# lambda - wavelength of the sunusoidal factor
	# gamma - spatial aspect ratio
	# psi - phase offset
	# ktype - type and range of values that each pixel in the gabor kernel can hold
	g_kernels = []
	for k in range (1,6):
		for i in range(1,9):
			g_kernel = cv2.getGaborKernel((32, 32), 8.0, (np.pi * i)/8, 2**k, 0.5, 0, ktype=cv2.CV_32F)
			g_kernels.append(g_kernel)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	filtered_imgs = []
	for g_kernel in g_kernels:
		filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
		filtered_imgs.append(filtered_img)
		#cv2.imshow('image', g_kernel)
		#cv2.imshow('filtered image', filtered_img)
		h, w = g_kernel.shape[:2]
		g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
		#cv2.imshow('image', g_kernel)
	gabor_features = []
	for image in filtered_imgs:
		gabor_features.append((get_amplitude(image)))
	return gabor_features