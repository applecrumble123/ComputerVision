import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math



#load image
img = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.1P/Resources_2.1/empire.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(img_gray, 'gray')
plt.show()

#kernel K defined above
avg_kernel = np.ones((5,5), np.float32) / 25
#always set the second parameter to -1, automatically calculate the dimension of image
avg_result = cv.filter2D(img_gray, -1, avg_kernel)
plt.imshow(avg_result, 'gray')
cv.imwrite('k_kernel.jpg',avg_result )
#plt.show()


#Gaussian Kernel
gaussian_kernel = np.float32([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])/256
gaussian = cv.filter2D(img_gray, -1, gaussian_kernel)
plt.imshow(gaussian, 'gray')
cv.imwrite('gaussian_kernel.jpg',gaussian)
#plt.show()


#Sobel Kernel
sobel_kernel = np.float32([[-1,0,1], [-2,0,2], [-1,0,1]])/8
sobel = cv.filter2D(img_gray, -1, sobel_kernel)
plt.imshow(sobel, 'gray')
cv.imwrite('sobel_kernel.jpg',sobel)
reverse_sobel_kernel = 255 - sobel
cv.imwrite('reverse_sobel_kernel.jpg',reverse_sobel_kernel)
#lt.show()


#Corner Kernel
corner_kernel = np.float32([[1,-2,1], [-2,4,-2], [1,-2,1]])/4
corner = cv.filter2D(img_gray, -1, corner_kernel)
plt.imshow(corner, 'gray')
cv.imwrite('corner_kernel.jpg',corner)
reverse_corner_kernel = 255 - sobel
cv.imwrite('reverse_corner_kernel.jpg',reverse_corner_kernel)
#plt.show()


#Testing median filter
img_noise = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.1P/Resources_2.1/empire_shotnoise.jpg')
img_noise_gray = cv.cvtColor(img_noise, cv.COLOR_BGR2GRAY)
ksize = 5 # neighbourhood of ksize x ksize; ksize must be an odd number
med_result = cv.medianBlur(img_noise_gray, ksize)
plt.imshow(med_result, 'gray')
cv.imwrite('median_result.jpg',med_result)
#plt.show()


#Testing bilateral filter
#radius to determine neighbourhood
rad = 5
#standard deviation for spatial distance (see slide 21 in week 2 lecture slides)
sigma_s = 10
#standard deviation for colour difference (see slide 21 in week 2 lecture slides)
sigma_c = 30
bil_result = cv.bilateralFilter(img_noise_gray, rad, sigma_c, sigma_s)
plt.imshow(bil_result, 'gray')
cv.imwrite('bilateral_result.jpg',bil_result)
#plt.show()

# Gaussian filter on img_noise_gray
gaussian_noise = cv.filter2D(img_noise_gray, -1, gaussian_kernel)
plt.imshow(gaussian_noise, 'gray')
cv.imwrite('gaussian_kernel_noise.jpg',gaussian_noise)
#plt.show()


# Edge Detection using sobel kernels
D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
der_x = cv.filter2D(img_gray, -1, D_x)
plt.imshow(der_x, 'gray')

D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
der_y = cv.filter2D(img_gray, -1, D_y)
plt.imshow(der_y, 'gray')

# height = row
# width = col
height, width = img_gray.shape
#gradient magnitude of img_gray

# initialise the array
mag_img_gray = np.zeros((height, width), np.float32)

# append each calculation into mag_img_gray
# range of index for height
for i in range(0, height):
    # range of index for width
    for j in range(0, width):
        square_der_x = float(der_x[i, j]) * float(der_x[i, j])
        square_der_y = float(der_y[i, j]) * float(der_y[i, j])
        mag_img_gray[i, j] = int(math.sqrt(square_der_x + square_der_y))

plt.imshow(mag_img_gray,'gray')
cv.imwrite('sobel_edges.jpg', mag_img_gray)
reverse_sobel = 255 - mag_img_gray
plt.imshow(reverse_sobel, 'gray')
cv.imwrite('reverse_sobel_edges.jpg', reverse_sobel)
plt.show()

# Edges for canny edge detector
minVal = 100 #minVal used in hysteresis thresholding
maxVal = 200 #maxVal used in hysteresis thresholding
Canny_edges = cv.Canny(img_gray, minVal, maxVal)
plt.imshow(Canny_edges, 'gray')
cv.imwrite('canny_edges.jpg', Canny_edges)
reverse_canny = 255 - Canny_edges
plt.imshow(reverse_canny, 'gray')
cv.imwrite('reverse_canny_edges.jpg', reverse_canny)
plt.show()