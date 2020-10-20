import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer


"""
The Dictionary class below is developed to build BoW models using K-means algorithm.  
K- means is used and the create_word_histograms method which constructs word histograms for a given list of images.
"""
class Dictionary(object):
    # initialise the class
    def __init__(self, name, img_filenames, num_words):
        self.name = name #name of your dictionary
        self.img_filenames = img_filenames #list of image filenames
        self.num_words = num_words #the number of words

        self.training_data = [] #this is the training data required by the K-Means algorithm
        self.words = [] #list of words, which are the centroids of clusters

    def learn(self):
        sift = cv.xfeatures2d.SIFT_create()

        num_keypoints = []  # this is used to store the number of keypoints in each image

        # load training images and compute SIFT descriptors
        for filename in self.img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            list_des = sift.detectAndCompute(img_gray, None)[1]

            # if there is not descriptors (such as a white piece of paper), there are 0 key points
            if list_des is None:
                num_keypoints.append(0)
            else:
                num_keypoints.append(len(list_des))
                for des in list_des:
                    self.training_data.append(des)

        # cluster SIFT descriptors using K-means algorithm
        kmeans = KMeans(self.num_words)
        kmeans.fit(self.training_data)

        # each word represent a cluster centroid
        self.words = kmeans.cluster_centers_

        # create word histograms for training images
        training_word_histograms = []  # list of word histograms of all training images
        index = 0
        for i in range(0, len(self.img_filenames)):
            # for each file, create a histogram
            histogram = np.zeros(self.num_words, np.float32)

            # if some keypoints exist
            if num_keypoints[i] > 0:
                for j in range(0, num_keypoints[i]):
                    histogram[kmeans.labels_[j + index]] += 1
                index += num_keypoints[i]
                histogram /= num_keypoints[i]
                training_word_histograms.append(histogram)

        return training_word_histograms

    def create_word_histograms(self, img_filenames):
        sift = cv.xfeatures2d.SIFT_create()
        histograms = []

        for filename in img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            descriptors = sift.detectAndCompute(img_gray, None)[1]

            histogram = np.zeros(self.num_words, np.float32)  # word histogram for the input image

            if descriptors is not None:
                for des in descriptors:
                    # find the best matching word
                    min_distance = 1111111  # this can be any large number
                    matching_word_ID = -1 #initial matching_word_ID=-1 means no matching

                    # search for the best matching word
                    for i in range(0,self.num_words):
                        distance = np.linalg.norm(des - self.words[i])
                        if distance < min_distance:
                            min_distance = distance
                            matching_word_ID = i

                    histogram[matching_word_ID] += 1

                histogram /= len(descriptors)  # normalise histogram to frequencies

            histograms.append(histogram)

        return histograms

import os

"""
training_file_names: containing the file names of all training images
training_food_labels: containing the food labels of all training images
Cakes have labels as 0, Pasta as 1, and Pizza as 2.
"""

foods = ['Cakes', 'Pasta', 'Pizza']
path = '/Users/johnathontoh/Desktop/SIT789 - Applications of Computer Vision and Speech Processing/Week 4/Task 4.1P/Resources_4.1/FoodImages'
training_file_names = []
training_food_labels = []
for i in range(0, len(foods)):
    sub_path = path + '/Train/' + foods[i] + '/'
    print("sub_path: \n", sub_path, "\n")

    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    print("sub_file_names: \n", sub_file_names, "\n")

    # create a list of N elements, all are i
    training_file_names += sub_file_names
    print("training_file_names: \n", training_file_names, "\n")

    sub_food_labels = [i] * len(sub_file_names)
    print("sub_food_labels: \n", sub_food_labels, "\n")

    training_food_labels += sub_food_labels
    print("training_food_labels: \n", training_food_labels, "\n")


print("training_file_names: \n", training_file_names, "\n")
print("training_food_labels: \n",training_food_labels, "\n")


# 50 words in the dictionary
num_words = 50
dictionary_name = 'food'
dictionary = Dictionary(dictionary_name, training_file_names, num_words)




"""
The learn method extracts words from a training dataset and creates the word histograms for all the training images in the training set.
"""
training_word_histograms = dictionary.learn()

"""
Save the dictionary into file once the training is complete and reload it for use without retraining. 
To save the dictionary into file, you can use pickle as follows.
"""
import pickle
#save dictionary
with open('food_dictionary.dic', 'wb') as f: #'wb' is for binary write
    pickle.dump(dictionary, f)

# load the dictionary
with open('food_dictionary.dic', 'rb') as f: #'rb' is for binary read
    dictionary = pickle.load(f)
    #print(dictionary)

"""
apply k-NN for food image recognition. 
First declare a k-NN classifier and train it using the training_word_histograms and training_food_labels.
"""

def knn(classifier, num ):
    num_nearest_neighbours = num #number of neighbours
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
    knn.fit(training_word_histograms, training_food_labels)

    predicted_food_labels_array = []

    for folder in os.listdir(path + '/Test'):
        # to not count any hidden files
        if folder.startswith('.') is False:
            for files in os.listdir(path + '/Test/' + folder):
                test_file_name = [path + '/Test/' + folder + '/' + files]
                print(test_file_name)
                word_histograms = dictionary.create_word_histograms(test_file_name)

                predicted_food_labels = knn.predict(word_histograms)
                predicted_food_labels_array.append(predicted_food_labels)
                #print('Food label: ', predicted_food_labels)
    print("predicted_food_labels_array: \n", predicted_food_labels_array, "\n")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(training_food_labels, predicted_food_labels_array)
    print(cm)
    correct_predictions = cm[0][0] + cm[1][1] + cm[2][2]
    print("The number of correct prediction is {}".format(correct_predictions))
    total_predictions = 0
    for i in cm:
        sum_of_array =i[0] + i[1] + i[2]
        total_predictions = total_predictions + sum_of_array
    print("The total number of images is {}".format(total_predictions))

    recognition_accuracy = (correct_predictions/total_predictions)*100
    print("The recognition accuracy for {} is {:.2f}% for {} nearest neighbours\n".format(classifier,recognition_accuracy, num))



def SVM(classifier, num):
    from sklearn import svm
    svm_classifier = svm.SVC(C=num,  # see slide 32 in week 4 lecture slides
                             kernel='linear')  # see slide 35 in week 4 lecture slides
    svm_classifier.fit(training_word_histograms, training_food_labels)

    predicted_food_labels_array = []

    for folder in os.listdir(path + '/Test'):
        # to not count any hidden files
        if folder.startswith('.') is False:
            for files in os.listdir(path + '/Test/' + folder):
                test_file_name = [path + '/Test/' + folder + '/' + files]
                word_histograms = dictionary.create_word_histograms(test_file_name)

                predicted_food_labels = svm_classifier.predict(word_histograms)
                predicted_food_labels_array.append(predicted_food_labels)
                #print('Food label: ', predicted_food_labels)
    print(predicted_food_labels_array)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(training_food_labels, predicted_food_labels_array)
    print(cm)
    correct_predictions = cm[0][0] + cm[1][1] + cm[2][2]
    print("The number of correct prediction is {}".format(correct_predictions))
    total_predictions = 0
    for i in cm:
        sum_of_array =i[0] + i[1] + i[2]
        total_predictions = total_predictions + sum_of_array
    print("The total number of images is {}".format(total_predictions))

    recognition_accuracy = (correct_predictions/total_predictions)*100
    print("The recognition accuracy for {} is {:.2f}% for {} C\n".format(classifier,recognition_accuracy, num))


def adaboost(classifier, num):
    from sklearn.ensemble import AdaBoostClassifier
    adb_classifier = AdaBoostClassifier(n_estimators=num,  # weak classifiers
                                        random_state=0)
    adb_classifier.fit(training_word_histograms, training_food_labels)

    predicted_food_labels_array = []

    for folder in os.listdir(path + '/Test'):
        # to not count any hidden files
        if folder.startswith('.') is False:
            for files in os.listdir(path + '/Test/' + folder):
                test_file_name = [path + '/Test/' + folder + '/' + files]
                word_histograms = dictionary.create_word_histograms(test_file_name)

                predicted_food_labels = adb_classifier.predict(word_histograms)
                predicted_food_labels_array.append(predicted_food_labels)
                #print('Food label: ', predicted_food_labels)
    print(predicted_food_labels_array)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(training_food_labels, predicted_food_labels_array)
    print(cm)
    correct_predictions = cm[0][0] + cm[1][1] + cm[2][2]
    print("The number of correct prediction is {}".format(correct_predictions))
    total_predictions = 0
    for i in cm:
        sum_of_array =i[0] + i[1] + i[2]
        total_predictions = total_predictions + sum_of_array
    print("The total number of images is {}".format(total_predictions))

    recognition_accuracy = (correct_predictions/total_predictions)*100
    print("The recognition accuracy for {} is {:.2f}% for {} n_estimators\n".format(classifier,recognition_accuracy, num))


knn('KNN',5)
#knn('KNN',10)
#knn('KNN',15)
#knn('KNN',20)
#knn('KNN',25)
#knn('KNN',30)

#SVM('SVM', 10)
#SVM('SVM', 20)
#SVM('SVM', 30)
#SVM('SVM', 40)
#SVM('SVM', 50)

#adaboost('AdaBoost', 50)
#adaboost('AdaBoost', 100)
#adaboost('AdaBoost', 150)
#adaboost('AdaBoost', 200)
#adaboost('AdaBoost', 250)

