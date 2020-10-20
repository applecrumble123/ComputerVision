import numpy as np
import cv2 as cv
import math
import statistics
from matplotlib import pyplot as plt

def edge_orientation(name, img_file):
    #load image
    img = cv.imread(img_file)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    der_x = cv.filter2D(img_gray, -1, D_x)

    D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
    der_y = cv.filter2D(img_gray, -1, D_y)


    height, width = img_gray.shape
    #gradient orientation of img_gray
    ori_img_gray = np.zeros((height, width), np.float32)

    # range of index for height
    for i in range(0, height):
        # range of index for width
        for j in range(0, width):
            if der_x[i, j] == 0 and der_y[i, j] == 0:
                ori_img_gray[i, j] = math.inf
            else:
                ori_img_gray[i, j] = math.atan2(der_y[i, j], der_x[i, j])

                # edge orientations are orthogonal to the gradient directions
                # add 90 to the gradient directions of the pixels
                #ori_img_gray[i, j] = ori_img_gray[i, j] + 90
            ori_img_gray[i, j] = (ori_img_gray[i, j] * 180)/math.pi
            ori_img_gray[i, j] = ori_img_gray[i, j] + 90
            #print(ori_img_gray[i, j])

    hist1_gray = cv.calcHist([ori_img_gray],[0],None,[181],[0,181])
    plt.plot(hist1_gray, label = name)
    plt.legend(loc="upper right")
    plt.xlim([0,181])
    plt.show()

edge_orientation('fisherman','/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.2C/Resources_2.2/fisherman.jpg')
edge_orientation('empire','/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.2C/Resources_2.2/empire.jpg')




def morphology(closing_operation, img_file):
    # Applying morphology for document skew estimation
    # Note that the second parameter of imread is set to 0
    doc = cv.imread(img_file, 0)

    # intensity value for each pixel is integer and ranges in [0, 255] where 0 represents BLACK and 255 represents WHITE
    threshold = 200
    ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)

    # 1 row 15 col to structure the lines on the document
    #structuring element for closing
    closing_se = closing_operation

    """
    morphological transforms are applied on FOREGROUND pixels. 
    By default, OpenCV considers WHITE pixels as foreground pixels while the text in our image doc_bin is presented in BLACK.
    """
    #convert black to white and white to black
    doc_bin = 255 - doc_bin

    # apply closing operation on doc_bin.
    closing = cv.morphologyEx(doc_bin, cv.MORPH_CLOSE, closing_se)
    plt.imshow(closing, 'gray')
    plt.show()

    """
    The closing operation aims to link letters with the same words and words within the same text lines. 
    However, it may also link text from different text lines and/or background noise. 
    To remove such unexpected links, opening operation is applied on the result of closing operation subsequently.
    We first define a structuring element for opening operation.
    """
    #structuring element for opening
    opening_se = np.ones((8, 8), np.int)

    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, opening_se)
    plt.imshow(opening, 'gray')
    plt.show()

    # Extract merged text lines using connected component labelling technique where each text line corresponds to a connected component.
    # connected component labelling
    num_labels, labels_im = cv.connectedComponents(opening)

    """
    cv.connectedComponents receives input as a binary image and groups WHITE pixels into connected components. 
    The output of cv.connectedComponents includes the number of connected components and an
    image containing labels of connected components. 
    In the output labelled image, e.g., labels_im above, each foreground pixel is labelled with the ID of a connected component it belongs to.
    we transform the output of cv.connectedComponents into an array structure where each element is a list of pixels of a connected component. 
    The following function is used for such purpose.
    """

    def ExtractConnectedComponents(num_labels, labels_im):
        connected_components = [[] for i in range(0, num_labels)]

        height, width = labels_im.shape
        for i in range(0, height):
            for j in range(0, width):
                if labels_im[i, j] >= 0:
                    connected_components[labels_im[i, j]].append((j, i))
        return connected_components


    connected_components = ExtractConnectedComponents(num_labels, labels_im)

    def FindOrientation(cc):
        mx = 0
        my = 0
        mxx = 0
        myy = 0
        mxy = 0
        for i in range(0, len(cc)):
            # cc[i][0] is used to store the x coordinate of pixel cc[i]
            mx += cc[i][0]
            # cc[i][1] is used to store the y coordinate of pixel cc[i]
            my += cc[i][1]
        mx /= len(cc)
        my /= len(cc)

        for i in range(0, len(cc)):
            dx = cc[i][0] - mx
            dy = cc[i][1] - my
            mxx += (dx * dx)
            myy += (dy * dy)
            mxy += (dx * dy)
        mxx /= len(cc)
        myy /= len(cc)
        mxy /= len(cc)

        theta = - math.atan2(2 * mxy, mxx - myy) / 2
        return theta

    """
    We now call FindOrientation for all connected components and 
    stores resulting orientations in a array named "orientations" for further processing
    """

    orientations = np.zeros(num_labels, np.float32)
    for i in range(0, num_labels):
        orientations[i] = FindOrientation(connected_components[i])

    """
    The orientation of the entire document is computed as the median of the orientations of all text lines. 
    We will call median method from statistics to get this median.
    """
    orientation = statistics.median(orientations)



    """
    We now deskew the image in doc by rotating it with an angle of -orientation.
    """

    # rotate image
    height, width = doc.shape
    # column index varies in [0, width-1]
    c_x = (width - 1) / 2.0
    # row index varies in [0, height-1]
    c_y = (height - 1) / 2.0
    # A point is defined by x and y coordinate
    c = (c_x, c_y)
    M = cv.getRotationMatrix2D(c, -orientation * 180 / math.pi, 1)
    doc_deskewed = cv.warpAffine(doc, M, (width, height))
    plt.imshow(doc_deskewed, 'gray')
    plt.show()

morphology(np.ones((1, 15), np.int),'/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.2C/Resources_2.2/doc.jpg')
morphology(np.ones((15, 1), np.int),'/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.2C/Resources_2.2/doc_1.jpg')