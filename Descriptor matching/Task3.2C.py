import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#load images
img = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.2C/Resources_3.2/empire.jpg')
img_45 = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.2C/Resources_3.2/empire_45.jpg')
img_zoomedout = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.2C/Resources_3.2/empire_zoomedout.jpg')
img_another = cv.imread('/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 3/Task 3.2C/Resources_3.2/fisherman.jpg')

#convert the images to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_45_gray = cv.cvtColor(img_45, cv.COLOR_BGR2GRAY)
img_zoomedout_gray = cv.cvtColor(img_zoomedout, cv.COLOR_BGR2GRAY)
img_another_gray = cv.cvtColor(img_another, cv.COLOR_BGR2GRAY)

def descriptor_matching(num_of_matches, img_gray_original, img_other, img1, img2):
    #initialise SIFT
    sift = cv.xfeatures2d.SIFT_create()

    #extract keypoints and descriptors
    kp_original, des_original = sift.detectAndCompute(img_gray_original, None)
    kp_other, des_other = sift.detectAndCompute(img_other, None)

    # Initialise a brute force matcher with default params
    """
    These sets of descriptors are called query and train. For each descriptor in the query, the BFM finds its best match in train. 
    The matching score between two descriptors is measured by the distance, e.g., Euclidean distance, between the descriptors. 
    For instance, to find matching descriptors of des on des_45
    """
    # For each descriptor in the query, the BFM finds its best match in train
    bf = cv.BFMatcher()


    # ----- for each query in the original image (query), finds the best descriptor in the the other image (train) --------
    train = des_other
    query = des_original
    matches_des_des_other = bf.match(query, train)
    #  sort the matches based on their matching scores using distance
    matches_des_des_other = sorted(matches_des_des_other, key = lambda x:x.distance)
    # Draw the best number of matches.
    # Draw the best 10 matches.
    nBestMatches = num_of_matches
    matching_des_des_other = cv.drawMatches(img_gray_original, kp_original, img_other, kp_other, matches_des_des_other[:nBestMatches], None,
                                            flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matching_des_des_other)
    plt.show()

    kp_train = kp_other
    kp_query = kp_original

    distance1 = 0
    for i in range (0, nBestMatches):
        print("match ", i, " info")

        print("\tdistance:", matches_des_des_other[i].distance)

        print("\tkeypoint in train: ID:", matches_des_des_other[i].trainIdx,
              " x:", kp_train[matches_des_des_other[i].trainIdx].pt[0],
              " y:", kp_train[matches_des_des_other[i].trainIdx].pt[1])

        print("\tkeypoint in query: ID:", matches_des_des_other[i].queryIdx,
              " x:", kp_query[matches_des_des_other[i].queryIdx].pt[0],
              " y:", kp_query[matches_des_des_other[i].queryIdx].pt[1])
        distance1 = distance1 + matches_des_des_other[i].distance

    print("\nDistance between {} and {} is {:.3f} \n".format(img1, img2, distance1))



    # ----- for each query in the other image (query), finds the best descriptor in the the original image (train) --------
    train2 = des_original
    query2 = des_other
    matches_des_other_des = bf.match(query2, train2)
    matches_des_other_des = sorted(matches_des_other_des, key = lambda x:x.distance)
    matching_des_other_des = cv.drawMatches(img_other, kp_other, img_gray_original, kp_original, matches_des_other_des[:nBestMatches], None,
                                         flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matching_des_other_des)
    plt.show()

    kp_train2 = kp_original
    kp_query2 = kp_other

    distance2 = 0
    for i in range (0, nBestMatches):
        print("match2 ", i, " info2")

        print("\tdistance2:", matches_des_other_des[i].distance)

        print("\tkeypoint in train2: ID:", matches_des_other_des[i].trainIdx,
              " x:", kp_train2[matches_des_other_des[i].trainIdx].pt[0],
              " y:", kp_train2[matches_des_other_des[i].trainIdx].pt[1])

        print("\tkeypoint in query2: ID:", matches_des_other_des[i].queryIdx,
              " x:", kp_query2[matches_des_other_des[i].queryIdx].pt[0],
              " y:", kp_query2[matches_des_other_des[i].queryIdx].pt[1])
        distance2 = distance2 + matches_des_other_des[i].distance

    print("\nDistance between {} and {} is {:.3f}\n".format(img2, img1, distance2))

    similarity_distance = (distance1 + distance2)/2
    print("Similarity Distance between {} and {} is {:.3f}\n".format(img1, img2, similarity_distance))

#descriptor_matching(10 ,img_gray, img_45_gray, "img_gray", "img_45_gray")
#descriptor_matching(30 ,img_gray, img_45_gray, "img_gray", "img_45_gray")
#descriptor_matching(10 ,img_gray, img_zoomedout_gray, "img_gray", "img_zoomedout_gray")
#descriptor_matching(30 ,img_gray, img_zoomedout_gray, "img_gray", "img_zoomedout_gray")
#descriptor_matching(10 ,img_gray, img_another_gray, "img_gray", "img_another_gray")
descriptor_matching(30 ,img_gray, img_another_gray, "img_gray", "img_another_gray")