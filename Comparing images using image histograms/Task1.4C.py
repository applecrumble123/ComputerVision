import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

#  read images and convert them to grayscale
img1 = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.4C/Resources_1.4/img1.jpg')
img2 = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.4C/Resources_1.4/img2.jpg')
img3 = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 1/Task 1.4C/Resources_1.4/img3.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img3_gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

# calculate the histograms of img1_gray, img2_gray, and img3_gray
hist1_gray = cv.calcHist([img1_gray],[0],None,[256],[0,256])
hist2_gray = cv.calcHist([img2_gray],[0],None,[256],[0,256])
hist3_gray = cv.calcHist([img3_gray],[0],None,[256],[0,256])


new_hist1_gray = np.array(hist1_gray).reshape(256,)
new_hist2_gray = np.array(hist2_gray).reshape(256,)
new_hist3_gray = np.array(hist3_gray).reshape(256,)


plt.plot(hist1_gray, label = 'img1_gray')
plt.plot(hist2_gray, label = 'img2_gray')
plt.plot(hist3_gray, label = 'img3_gray')

plt.legend(loc="upper right")
plt.xlim([0,256])
plt.show()

def chi_square(hist1, hist2, img1, img2):
    chi_square_img = 0
    for i in range(0, 256):
        if (hist1[i] + hist2[i]) > 0:
            indvidual_cal = (math.pow(hist1[i] - hist2[i], 2)) / (hist1[i] + hist2[i])
            # print(indvidual_cal)
            chi_square_img = chi_square_img + indvidual_cal
        else:
            indvidual_cal = (math.pow(hist1[i] - hist2[i], 2)) / (hist1[i] + hist2[i] + 0.000001)
            # print(indvidual_cal)
            chi_square_img = chi_square_img + indvidual_cal
    print("chi square distance between {} and {} is {}".format(img1, img2, chi_square_img))

chi_square(new_hist1_gray, new_hist2_gray, "img1", "img2")
chi_square(new_hist1_gray, new_hist3_gray, "img1", "img3")
chi_square(new_hist2_gray, new_hist3_gray, "img2", "img3")


def Kullback_Leibler_divergence(hist1, hist2, img1, img2):
    total_h1 = 0
    for i in range(0, 256):
        total_h1 = total_h1 + hist1[i]

    total_h2 = 0
    for i in range(0, 256):
        total_h2 = total_h2 + hist2[i]

    KL_img1_img2 = 0
    KL_img2_img1 = 0
    for i in range (0, 256):
        KL_h1 = (hist1[i]) / total_h1
        if KL_h1 == 0:
            KL_h1 = KL_h1 + 0.000001

        KL_h2 = (hist2[i]) / total_h2
        if KL_h2 == 0:
            KL_h2 = KL_h2 + 0.000001

        KL_h1_h2 = KL_h1/KL_h2
        KL_indvidual_cal = KL_h1 * (math.log2(KL_h1_h2))
        KL_img1_img2 = KL_img1_img2 + KL_indvidual_cal

        KL_h2_h1 = KL_h2 / KL_h1
        KL_indvidual_cal2 = KL_h2 * (math.log2(KL_h2_h1))
        KL_img2_img1 = KL_img2_img1 + KL_indvidual_cal2

    total_KL_distance = KL_img1_img2 + KL_img2_img1

    print("\nKL distance between {} and {} is {}".format(img1, img2, KL_img1_img2))
    print("KL distance between {} and {} is {}".format(img2, img1, KL_img2_img1))
    print("Combined KL distance for {} and {} is {}".format(img1, img2, total_KL_distance))



Kullback_Leibler_divergence(new_hist1_gray, new_hist2_gray, "img1", "img2")
Kullback_Leibler_divergence(new_hist1_gray, new_hist3_gray, "img1", "img3")
Kullback_Leibler_divergence(new_hist2_gray, new_hist3_gray, "img2", "img3")

