import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.3C/img1.jpg')

#[0] for blue channel
# Green and red channel can be set to [1] and [2] respectively.
hist_blue = cv.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist_blue, label = 'blue channel', color = 'b')
plt.legend(loc="upper left")
plt.xlim([0,256])

# Green channel
hist_green = cv.calcHist([img],[1],None,[256],[0,256])
plt.plot(hist_green, label = 'green channel', color = 'g')
plt.legend(loc="upper left")
plt.xlim([0,256])

# Red channel
hist_red = cv.calcHist([img],[2],None,[256],[0,256])
plt.plot(hist_red, label = 'red channel', color = 'r')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()

# grayscale is used for intensity
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist_gray = cv.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(hist_gray, label = 'intensity',)
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()

# summation of all the calculations
def getCummulativeDis(hist):
    c = [] #cummulative distribution
    s=0
    for i in range(0, len(hist)):
        s = s + hist[i]
        c.append(s)
    return c

# it shows that there are no equalisation as there is no straight line
c = getCummulativeDis(hist_gray)
plt.plot(c, label = 'initial cummulative distribution', color = 'r')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()

"""
cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".

channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. 
            For example, if input is grayscale image, its value is [0]. 
            For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.

mask : mask image. To find histogram of full image, it is given as "None". 
        But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask.

histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].

ranges : this is our RANGE. Normally, it is [0,256].
"""

# apply histogram euqualisation on img_gray
img_equ = cv.equalizeHist(img_gray)
hist_equ = cv.calcHist([img_equ],[0],None,[256],[0,256])
plt.plot(hist_equ, label = 'histogram euqualisation on intensity' )
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()

# calculate and plot the cummulative distribution of intensity of img_equ
c_equ = getCummulativeDis(hist_equ)
plt.plot(c_equ, label = 'cummulative distribution after histogram equalisation', color = 'r')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()

#stacking images side-by-side
img_equalisation = np.hstack((img_gray, img_equ))
#writing the stacked image to file
cv.imwrite('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.3C/img_equalisation.png', img_equalisation)