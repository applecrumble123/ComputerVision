import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def harris_corner_detector(num, img):
    #load image
    img = cv.imread(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(img_gray, 'gray')

    #size of local image regions W introduced in slide 6 in week 3 slides
    local_region_size = 3
    #kernel size of Sobel kernels used for calculating derivatives I_x and I_y
    kernel_size = 3
    #parameter k in side 6 in week 3 slides
    k = 0.04
    #threshold theta introduced in slide 6 in week 3 slides
    threshold = 1000.0

    #img_gray contains integer values (for pixel intensity) and therefore needs to be converted to float before applying Harris corner detector.
    img_gray = np.float32(img_gray)

    Harris_res_img = cv.cornerHarris(img_gray, local_region_size, kernel_size, k)
    plt.imshow(Harris_res_img, 'gray')
    plt.show()

    highlighted_colour = [0, 0, 255] # a colour is a combination of blue, green, red; red=[0,0,255]
    highlighted_img = img.copy()
    highlighted_img[Harris_res_img > threshold] = highlighted_colour

    # colour is reversed
    plt.imshow(highlighted_img[:,:,::-1]) # RGB-> BGR
    plt.show()

    height, width = Harris_res_img.shape
    print(Harris_res_img.shape)

    count = 0
    for i in range(0, height):
        for j in range(0, width):
            if Harris_res_img[i, j] > 1000:
                count = count + 1

    print("Count of pixels more than the threshold (1000) is {}".format(count))

    new_threshold = num * Harris_res_img.max()
    new_highlighted_colour = [0, 255, 0]  # a colour is a combination of blue, green, red; red=[0,0,255]
    new_highlighted_img = img.copy()

    new_highlighted_img[Harris_res_img > new_threshold] = new_highlighted_colour

    # colour is reversed
    plt.imshow(new_highlighted_img[:, :, ::-1])  # RGB-> BGR
    plt.show()

#harris_corner_detector(0.001, '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire.jpg')
#harris_corner_detector(0.005, '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire.jpg')
#harris_corner_detector(0.02, '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire.jpg')
#harris_corner_detector(0.07, '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire.jpg')

# opencv-python==3.4.2.17
# opencv-contrib-python==3.4.2.17
sift = cv.xfeatures2d.SIFT_create()


img = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
kp = sift.detect(img_gray, None)

# make copy of the image
img_gray_kp = img_gray.copy()
# To visualise SIFT keypoints, you can create an image which is identical to img_gray and draw keypoints on this new image
img_gray_kp = cv.drawKeypoints(img_gray, kp, img_gray_kp)
plt.imshow(img_gray_kp)
#plt.show()
print("Number of detected keypoints: %d" % (len(kp)))

# visualise local image regions used for extracting SIFT descriptors
img_gray_kp = cv.drawKeypoints(img_gray, kp, img_gray_kp, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_gray_kp)
#plt.show()

# extract SIFT descriptors
# kp = keypoints
# description of the keypoints
kp, des = sift.compute(img_gray, kp)

"""
des is an array of size (2804x128). The rows correspond to keypoints and thus there are 2804 rows. 
The columns represent descriptors, each column store a descriptor including 128 elements.
"""
print(des.shape)


img_45 = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire_45.jpg')
img_45_gray = cv.cvtColor(img_45, cv.COLOR_BGR2GRAY)

img_zoomedout = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/empire_zoomedout.jpg')
img_zoomedout_gray = cv.cvtColor(img_zoomedout, cv.COLOR_BGR2GRAY)

img_another = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.1P/Resources_3.1/fisherman.jpg')
img_another_gray = cv.cvtColor(img_another, cv.COLOR_BGR2GRAY)

"""
both keypoint detection and descriptor computation can be done silmutaneously by calling sift.detectAndCompute.
"""
kp_45, des_45 = sift.detectAndCompute(img_45_gray, None)
kp_zoomedout, des_zoomedout = sift.detectAndCompute(img_zoomedout_gray, None)
kp_another, des_another = sift.detectAndCompute(img_another_gray, None)

kp, des = sift.compute(img_gray, kp)

print("The number of keypoints in img_gray is %d" % (len(des)))
print("The number of keypoints in img_45_gray is %d" % (len(des_45)))


#  Hausdorff distances
from scipy.spatial.distance import directed_hausdorff

# shorter distance means more similar
des_vs_des_45 = directed_hausdorff(des, des_45)[0]
print("des_vs_des_45 is {}".format(des_vs_des_45))

des_vs_des_zoomedout = directed_hausdorff(des, des_zoomedout)[0]
print("des_vs_des_zoomedout is {}".format(des_vs_des_zoomedout))

des_vs_des_another = directed_hausdorff(des, des_another)[0]
print("des_vs_des_another is {}".format(des_vs_des_another))