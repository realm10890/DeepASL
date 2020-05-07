import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import cv2
import sys
from os.path import isfile, join
from os import listdir, makedirs
from matplotlib import image
import glob, os, errno
from datetime import datetime
import os
from PIL import Image


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


#training labels an array of 1000 elements 500 0's and 500 1's [0 x 500, 1 x 500]
zeros = np.full((1, 500), 0)
ones = np.full((1, 500), 1)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()

train_labels = np.concatenate([finalZeros, finalOnes])

test_images = train_images[:1]
test_labels = train_labels[:1]


conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
#softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10
softmax = Softmax(13 * 13 * 8, 2)


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''

    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    # Outputs 8 convoled layers 26x26x8
    #print("Output Shape")
    # print(out.shape)

    out = pool.forward(out)
    # print()

    out = softmax.forward(out)
    # final output after softmax is a 10 index array with the decimal value of
    # what the network thinks is the number
    # the index of this output array correspond with the label 0-10

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])

    acc = 1 if np.argmax(out) == label else 0

    rightOrWrong = "Correct Prediction" if np.argmax(out) == label else "Incorrect Prediction"
    """""
    print("NEW SET")

    print("Predicted Output")
    print(np.argmax(out))

    print("Actual Output")
    print(label)

    print("Eval On Model Output")
    print(rightOrWrong)
    """""
    print("Predicted Output")
    print(np.argmax(out))
    print("Actual Output")
    print(label)
    print("Eval On Model Output")
    print(rightOrWrong)

    return out, loss, acc


def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    # TODO: backprop MaxPool2 layer
    # TODO: backprop Conv3x3 layer

    return loss, acc


print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(1000):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

    # Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print("num tests")
print(num_tests)
print("loss")
print(loss)
print('Test Loss:', loss / num_tests)
print("num correct ")
print(num_correct)
print('Test Accuracy:', num_correct / num_tests)

'''
loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass
    _, l, acc = forward(im, label)
    # this loss value is incremented by the output of the cross-entropy loss
    loss += l
    # num-correct is incremented by either 1 or 0 if the number was guessed
    # correctly then it is incremented by 1 else by 0
    num_correct += acc

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc
'''
