import cv2
import numpy as np
import sys
from os.path import isfile, join
from os import listdir, makedirs
from matplotlib import image
import glob, os, errno
from datetime import datetime
import os
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)

#training images taken each from a file and added into a numpy array(train_images)
train_images_asl_bnp = []
image_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_image_a_resized'
image_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_image_l_resized'

for filename in os.listdir(image_dir_a):
    if filename != ".DS_Store":
        readimg = cv2.imread(image_dir_a + '/' + filename)
        grayreadimg = cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
        train_images_asl_bnp.append(grayreadimg)

for filename2 in os.listdir(image_dir_l):
    if filename2 != ".DS_Store":
        readimg2 = cv2.imread(image_dir_l + '/' + filename2)
        grayreadimg2 = cv2.cvtColor(readimg2, cv2.COLOR_BGR2GRAY)
        train_images_asl_bnp.append(grayreadimg2)

train_images = np.array(train_images_asl_bnp)
print(train_images[700])

#training labels an array of 1000 elements 500 0's and 500 1's [0 x 500, 1 x 500]
zeros = np.full((1, 500), 0)
ones = np.full((1, 500), 1)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()
train_labels = np.concatenate([finalZeros, finalOnes])
print(train_labels[700])
