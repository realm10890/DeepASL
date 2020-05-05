import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# The mnist package takes care of handling the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
# We only use the first 1k testing examples (out of 10k total) in the interest of time.
# Feel free to change this if you want.
test_images = mnist.test_images()[:1000]


test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


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
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    # TODO: backprop MaxPool2 layer
    # TODO: backprop Conv3x3 layer

    return loss, acc


print('MNIST CNN initialized!')

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
