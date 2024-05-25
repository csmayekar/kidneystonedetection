# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:25:43 2024

@author: COMPUTER
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import cv2
import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



basepath= os.path.normpath('Kidney Stone Disease Detection-100%')

def fd_hu_moments(image):
    #For Shape of signature Image
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def quantify_image(image):
    #For Speed and pressure of signature image

    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")

    # return the feature vector
    return features

def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    path=trainingPath
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
#        print(imagePath)
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # quantify the image
        features1 = quantify_image(image)
        features2 = fd_hu_moments(image)
        global_feature = np.hstack([features1,features2])

        # update the data and labels lists, respectively
        data.append(global_feature)
        
        labels.append(label)

    # return the data and labels
    return (np.array(data), np.array(labels))


trainingPath = os.path.sep.join([basepath, "training_set"])
testingPath = os.path.sep.join([basepath, "testing _set"])


## loading the training and testing data
#print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

def DT_Cl():
    
    # Initialize Decision Tree Classifier
    Decision = DecisionTreeClassifier(random_state=42)

    # Loop over the number of trials to run
    for i in range(0, 5): 
        
        # Train the model
        print("[INFO] training model {} of {}...".format(i + 1, 5))
        Decision.fit(trainX, trainY) 
    
        # Make predictions on the testing data
        predictions = Decision.predict(testX)
    
    # Save the trained model
    with open(basepath + '/clf_DT.pkl', 'wb') as f:
        pickle.dump(Decision, f)
            
    # Calculate accuracy and save model information
    accuracy = "DT Accuracy: {0:.2%}".format(accuracy_score(predictions, testY))
    model_saved = "DT Model Saved as <<clf_DT.pkl>>"
    
    result = accuracy + '\n' + model_saved
    print(result)
    return result

DT_Cl()