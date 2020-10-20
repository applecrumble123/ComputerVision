import numpy as np
import cv2 as cv

img = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/img1.jpg')


#----------------------------- colour conversion ------------------------------------
# image img is represented in BGR (Blue-Green-Red) space by default
# convert img into HSV space
# HSV means Hue-Saturation-Value
# Hue is the color.
# Saturation is the greyness, so that a Saturation value near 0 means it is dull or grey looking.
# Value is the brightness of the pixel
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('image in HSV', img_hsv)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgHSV.jpg', img_hsv)
# close all the windows when any key is pressed
cv.destroyAllWindows()

# convert img into gray image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('image in gray', img_gray)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgGRAY.jpg', img_gray)
cv.destroyAllWindows()

#----------------------------- scaling ------------------------------------
height, width = img.shape[:2]
# resize the image img by a horizontal scale of 0.5 and vertical scale of 0.4
h_scale = 0.5
v_scale = 0.4

# we need this as the new height must be interger
new_height = (int) (height * v_scale)

# we need this as the new width must be interger
new_width = (int) (width * h_scale)

img_resize = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR)

cv.imshow('resize', img_resize)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgSCALE.jpg', img_resize)
cv.destroyAllWindows()

#----------------------------- translation ------------------------------------

#  shifts an image to a new location determined by a translation vector
t_x = 100
t_y = 200

M = np.float32([[1, 0, t_x], [0, 1, t_y]])

#this will get the number of rows and columns in img
height, width = img.shape[:2]
img_translation = cv.warpAffine(img, M, (width, height))
cv.imshow('translation', img_translation)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgTRANSLATION.jpg', img_translation)
cv.destroyAllWindows()

#----------------------------- rotation ------------------------------------
#rotate 45 degrees in anti-clockwise
# -45 to rotate in clockwise
theta = 45

# column index varies in [0, width-1]
c_x = (width - 1) / 2.0

# row index varies in [0, height-1]
c_y = (height - 1) / 2.0

# A point is defined by x and y coordinate
c = (c_x, c_y)
print(c)

# s is the scale, when no scaling is done, scale = 1
s= 1
M = cv.getRotationMatrix2D(c, theta, s)

img_rotation = cv.warpAffine(img, M, (width, height))
cv.imshow('rotation', img_rotation)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgROTATION.jpg', img_rotation)
cv.destroyAllWindows()


#----------------------------- Affine ------------------------------------
m00 = 0.38
m01 = 0.27
m02 = -47.18
m10 = -0.14
m11 = 0.75
m12 = 564.32

# transformation matrix
M = np.float32([[m00, m01, m02], [m10, m11, m12]])

height, width = img.shape[:2]
img_affine = cv.warpAffine(img, M, (width, height))
cv.imshow('affine', img_affine)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/imgAFFINE.jpg', img_affine)
cv.destroyAllWindows()

# using cv.warpAffine to replace cv.resize
# h_scale wrt to x-axis
# v_scale wrt to y-axis
M = np.float32([[h_scale, 0, 0], [0, v_scale, 0]])
img_replace_resize_with_affine = cv.warpAffine(img, M, (width, height))
cv.imshow('img_replace_resize_with_affine', img_replace_resize_with_affine)
cv.waitKey(0)
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.2P/Resources_1.2/img_replace_resize_with_affine.jpg', img_replace_resize_with_affine)
cv.destroyAllWindows()