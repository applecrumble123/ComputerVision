import numpy as np
import cv2 as cv
import math
import statistics
from matplotlib import pyplot as plt
import time



def all_foreground_pixels(img):
    start = time.time()
    # Step 1
    doc = cv.imread(img, 0)
    # intensity value for each pixel is integer and ranges in [0, 255] where 0 represents BLACK and 255 represents WHITE
    threshold = 200
    ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)
    # Step 2
    doc_bin = 255 - doc_bin
    plt.imshow(doc_bin, "gray")
    plt.show()
    # Step 3
    num_labels, labels_im = cv.connectedComponents(doc_bin)
    def ExtractConnectedComponents(num_labels, labels_im):
        connected_components = [[] for i in range(0, num_labels)]
        height, width = labels_im.shape
        for i in range(0, height):
            for j in range(0, width):
                if labels_im[i, j] >= 0:
                    connected_components[labels_im[i, j]].append((j, i))
        return connected_components
    connected_components = ExtractConnectedComponents(num_labels, labels_im)
    # Step 6
    distance_resolution = 1
    angular_resolution = math.pi/180
    density_threshold = 10
    # Step 7
    lines = cv.HoughLines(doc_bin, distance_resolution, angular_resolution, density_threshold)
    #print(lines)
    # Step 8
    angles_array = []
    for i in range(len(lines)):
        distance, angle = lines[i][0]
        angle = angle * 180 / math.pi
        # find the alpha of the line corresponding to the x -axis
        angle = 90 - angle
        angles_array.append(angle)
        # print(angle)
    # Step 9
    median_angle = statistics.median(angles_array)
    print(median_angle)
    print(-median_angle)
    # rotate image
    height, width = doc.shape
    # column index varies in [0, width-1]
    c_x = (width - 1) / 2.0
    # row index varies in [0, height-1]
    c_y = (height - 1) / 2.0
    # A point is defined by x and y coordinate
    c = (c_x, c_y)
    M = cv.getRotationMatrix2D(c, -median_angle, 1)
    doc_deskewed = cv.warpAffine(doc, M, (width, height))
    end = time.time()
    time_taken = end - start
    print("Time taken is {:.2f}s".format(time_taken))
    plt.imshow(doc_deskewed, 'gray')
    plt.show()

def center_x_y_coordintates(img):
    start = time.time()
    doc = cv.imread(img, 0)
    # intensity value for each pixel is integer and ranges in [0, 255] where 0 represents BLACK and 255 represents WHITE
    threshold = 200
    ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)
    # Step 2
    doc_bin = 255 - doc_bin
    # Step 3
    num_labels, labels_im = cv.connectedComponents(doc_bin)
    def ExtractConnectedComponents(num_labels, labels_im):
        # gets the set of connected components
        connected_components = [[] for i in range(0, num_labels)]
        # set of images with ID of the connected component
        height, width = labels_im.shape
        for i in range(0, height):
            for j in range(0, width):
                # if the ID of the connected component in the image region is more or equals to 0
                if labels_im[i, j] >= 0:
                    # append the value of in the height and width array to the connected component
                    connected_components[labels_im[i, j]].append((j, i))
                    #print(connected_components)
        return connected_components
    connected_components = ExtractConnectedComponents(num_labels, labels_im)
    
    # using a list of connected components
    # initialise an array
    # number of connected components
    
    new_x_y = []
    for i in range(0, len(connected_components)):
        #print(connected_components[i])
    
        x = 0
        y = 0
        for j in range(0, len(connected_components[i])):
            # cc[i][0] is used to store the x coordinate of pixel cc[i]
            # adds on to the initialised x array
            #print(connected_components[i][j])
            x = x + connected_components[i][j][0]
            y = y + connected_components[i][j][1]
        #print(x)
        #print(y)
        mean_x = x / len(connected_components[i])
        mean_y = y / len(connected_components[i])
        #print(mean_x)
        #print(mean_y)
    
        new_x_y.append([int(mean_x), int(mean_y)])
    
    print(new_x_y)
    
    # doc_bin is fully black
    height, width = doc_bin.shape
    print(doc_bin.shape)
    for i in range(0, height):
        for j in range(0, width):
            doc_bin[i, j] = 0

    # convert black points to white points
    # image coor is x,y --> x changes the col and y changes the row --> col (x), row (y)
    # x coor affects the col, y coor affects the row
    for k in new_x_y:
        # matrics --> row, col
        doc_bin[k[1], k[0]] = 255


    plt.imshow(doc_bin, "gray")
    plt.show()
    
    distance_resolution = 1
    angular_resolution = math.pi/180
    density_threshold = 10
    # Step 7
    lines = cv.HoughLines(doc_bin, distance_resolution, angular_resolution, density_threshold)
    #print(lines)
    # Step 8
    angles_array = []
    for i in range(len(lines)):
        distance, angle = lines[i][0]
        angle = angle*180/math.pi
        # find the alpha of the line corresponding to the x -axis
        angle = 90 - angle
        angles_array.append(angle)
        #print(angle)
    print(angles_array)
    # Step 9
    median_angle = statistics.median(angles_array)
    print(median_angle)
    print(-median_angle)
    # rotate image
    height, width = doc.shape
    # column index varies in [0, width-1]
    c_x = (width - 1) / 2.0
    # row index varies in [0, height-1]
    c_y = (height - 1) / 2.0
    # A point is defined by x and y coordinate
    c = (c_x, c_y)
    # default is anti-clockwise, "-median" to rotate clockwise
    M = cv.getRotationMatrix2D(c, -median_angle, 1)
    doc_deskewed = cv.warpAffine(doc, M, (width, height))
    end = time.time()
    time_taken = end - start
    print("Time taken is {:.2f}s".format(time_taken))
    plt.imshow(doc_deskewed, 'gray')
    plt.show()



def max_y_coor(img):
    start = time.time()
    doc = cv.imread(img, 0)
    # intensity value for each pixel is integer and ranges in [0, 255] where 0 represents BLACK and 255 represents WHITE
    threshold = 200
    ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)
    # Step 2
    doc_bin = 255 - doc_bin
    # Step 3
    num_labels, labels_im = cv.connectedComponents(doc_bin)
    def ExtractConnectedComponents(num_labels, labels_im):
        # gets the set of connected components
        connected_components = [[] for i in range(0, num_labels)]
        # set of images with ID of the connected component
        height, width = labels_im.shape
        for i in range(0, height):
            for j in range(0, width):
                # if the ID of the connected component in the image region is more or equals to 0
                if labels_im[i, j] >= 0:
                    # append the value of in the height and width array to the connected component
                    connected_components[labels_im[i, j]].append((j, i))
        return connected_components
    connected_components = ExtractConnectedComponents(num_labels, labels_im)
    #print(connected_components)

    # initialised with the first connected component, first array, y-value
    max_value = connected_components[0][0][1]
    max_value_x = connected_components[0][0][0]
    new_x_y = []
    #print(len(connected_components))
    # number of connected components
    for i in range(0, len(connected_components)):
        print(connected_components[i])
        # find the max_y
        max_y = 0
        max_x = 0
        max_y_array = []
        # prints out the biggest element in each connected component
        for j in range(len(connected_components[i])):
            if connected_components[i][j][1] > max_y:
                max_y = connected_components[i][j][1]

        #print(max_y)

        for j in range(len(connected_components[i])):
            if connected_components[i][j][1] == max_y:
                max_y_array.append([connected_components[i][j][0], connected_components[i][j][1]])

        # get the max of x for max of y
        for k in max_y_array:
           if k[0] > max_x:
               max_x = k[0]
        #print(max_x)
        new_x_y.append([max_x, max_y])

    print(new_x_y)
    # doc_bin is fully black
    height, width = doc_bin.shape
    print(doc_bin.shape)

    for i in range(0, height):
        for j in range(0, width):
            doc_bin[i, j] = 0

    # convert black points to white points
    # image coor is x,y --> x changes the col and y changes the row --> col, row
    for k in new_x_y:
        # matrics --> row, col
        doc_bin[k[1], k[0]] = 255


    #print(doc_bin)
    plt.imshow(doc_bin, "gray")
    #plt.show()

    distance_resolution = 1
    angular_resolution = math.pi/180
    density_threshold = 10
    # Step 7
    lines = cv.HoughLines(doc_bin, distance_resolution, angular_resolution, density_threshold)
    #print(lines)
    # Step 8
    angles_array = []
    for i in range(0,len(lines)):
        distance, angle = lines[i][0]
        angle = angle * 180 / math.pi
        # find the alpha of the line corresponding to the x -axis
        angle = 90 - angle
        angles_array.append(angle)
        # print(angle)
    print(angles_array)
    # Step 9
    median_angle = statistics.median(angles_array)
    print(median_angle)
    print(-median_angle)

    # rotate image
    height, width = doc.shape
    # column index varies in [0, width-1]
    c_x = (width - 1) / 2.0
    # row index varies in [0, height-1]
    c_y = (height - 1) / 2.0
    # A point is defined by x and y coordinate
    c = (c_x, c_y)
    M = cv.getRotationMatrix2D(c, -median_angle, 1)
    doc_deskewed = cv.warpAffine(doc, M, (width, height))
    end = time.time()
    time_taken = end - start
    print("Time taken is {:.2f}s".format(time_taken))
    plt.imshow(doc_deskewed, 'gray')
    #plt.show()


#all_foreground_pixels('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc.jpg')
#center_x_y_coordintates('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc.jpg')
#max_y_coor('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc.jpg')

#all_foreground_pixels('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc_1.jpg')
#center_x_y_coordintates('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc_1.jpg')
#max_y_coor('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 2/Task 2.3D/doc_1.jpg')



