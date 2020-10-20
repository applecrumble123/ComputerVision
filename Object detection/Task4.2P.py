import cv2 as cv
import time
import os

"""
detect_face method receives input as a colour image and a cascade detector, 
then detects human faces from the input image, and returns a list of detected human faces (if any).
"""

path = '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 4/Task 4.2P/Resources_4.2/FaceImages/'

def detect_face(image, scale, neighbours, size, flags):
    start_time = time.time()
    image = cv.imread(image)
    #  pre-trained cascade detector
    cascade_detector = cv.CascadeClassifier('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 4/Task 4.2P/Resources_4.2/haarcascade_frontalface_default.xml')
    #convert input image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade_detector.detectMultiScale(image_gray,
                                                scaleFactor = scale, #the ratio between two consecutive scales
                                                minNeighbors = neighbours, #minimum number of overlapping windows to be considered
                                                minSize = size, #minimum size of detection window (in pixels)
                                                flags = flags) #scale the image rather than detection window
    #return faces
    print('Face detection is performed in %s seconds ---' % (time.time() - start_time))
    if (faces is not None):
        print('Found ', len(faces), ' faces')
    else:
        print('There is no face found!')

    from matplotlib import pyplot as plt
    for (x, y, w, h) in faces: #(x, y) are the coordinate of the topleft corner,
        # w, h are the width and height of the bounding box
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(image[:,:,::-1]) # RGB-> BGR
    plt.show()


#detect_face(path + 'img_1123.jpg', 1.2, 20, (50, 50), cv.CASCADE_SCALE_IMAGE)

def video_cam():
    def detect_face_function(image, cascade_detector):
        #convert input image to grayscale
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = cascade_detector.detectMultiScale(image_gray,
        scaleFactor = 1.1, #the ratio between two consecutive scales
        minNeighbors = 5, #minimum number of overlapping windows to be considered
        minSize = (30, 30), #minimum size of detection window (in pixels)
        flags = cv.CASCADE_SCALE_IMAGE) #scale the image rather than detection window
        return faces

    #OPENCV_VIDEOIO_PRIORITY_MSMF = 0
    #initialise webcam
    cam = cv.VideoCapture(0)
    #initialise cascade_detector
    cascade_detector = cv.CascadeClassifier('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 4/Task 4.2P/Resources_4.2/haarcascade_frontalface_default.xml')
    while True:
        #read the image from the cam
        _, image = cam.read()
        #detect human faces from the current image using the cascade_detector

        faces = detect_face_function(image, cascade_detector)
        #display detected faces
        for x, y, w, h in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), color = (0, 255, 0))
            cv.imshow('face detection demo', image)
        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break
    cam.release()

#video_cam()

import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

def detect_pedestrian_function (winstride, padding, scale_num):
    def nms(boxes):

        # We first convert boxes from list to array as required by non_max_suppression method
        # #In addition, each box in the array is encoded by the topleft and bottomright corners
        boxes_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        boxes_array = non_max_suppression(boxes_array, probs=None, overlapThresh=0.65)
        # create a new list of boxes to store results
        boxes_list = []
        for top_x, top_y, bottom_x, bottom_y in boxes_array: boxes_list.append(
            [top_x, top_y, bottom_x - top_x, bottom_y - top_y])
        return boxes_list

    def detect_pedestrian(image):
        #initialise the HOG descriptor and SVM classifier
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        image_resized = imutils.resize(image,
                                        width = min(400, image.shape[1])) #resize the input image so that

        scale = image.shape[1] / image_resized.shape[1]
        #detect pedestrians
        (boxes, _) = hog.detectMultiScale(image_resized,
                                          winStride= winstride,  # horizontal and vertical stride
                                          padding = padding,  #horizontal and vertical padding for each window
                                          scale = scale_num) #scale factor between two consecutive scales

        # non-maximum suppression
        boxes = nms(boxes)
        # resize the bounding boxes
        for box in boxes:
            box[0] = np.int(box[0] * scale)
            box[1] = np.int(box[1] * scale)
            box[2] = np.int(box[2] * scale)
            box[3] = np.int(box[3] * scale)

        return boxes

    image = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 4/Task 4.2P/Resources_4.2/PedestrianImages/person_029.png')
    start_time = time.time()
    pedestrians = detect_pedestrian(image)
    print('Pedestrian detection is performed in %s seconds ---' % (time.time() - start_time))


    from matplotlib import pyplot as plt
    if (pedestrians is not None):
        print('Found ', len(pedestrians), ' pedestrians')

        for (x, y, w, h) in pedestrians:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
        plt.imshow(image[:,:,::-1]) # RGB-> BGR
        plt.show()
    else:
        print('There is no pedestrian found!')

#detect_pedestrian_function ((2,2), (6,6), 1.05)
#detect_pedestrian_function ((4,4), (6,6), 1.05)
#detect_pedestrian_function ((6,6), (6,6), 1.05)
#detect_pedestrian_function ((8,8), (6,6), 1.05)

#detect_pedestrian_function ((4,4), (4,4), 1.05)
#detect_pedestrian_function ((4,4), (6,6), 1.05)
#detect_pedestrian_function ((4,4), (8,8), 1.05)
#detect_pedestrian_function ((4,4), (10,10), 1.05)

#detect_pedestrian_function ((4,4), (6,6), 1.05)
#detect_pedestrian_function ((4,4), (6,6), 1.1)
#detect_pedestrian_function ((4,4), (6,6), 1.15)
#detect_pedestrian_function ((4,4), (6,6), 1.2)