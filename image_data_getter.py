import cv2
import numpy as np


cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    _, frame = cap.read()

    #cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 58, 50], dtype= "uint8")
    upper_blue = np.array([30, 255, 255], dtype="uint8")

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    maskROI = mask[300 : 600, 300 : 600]


    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original Frame",frame)

    cv2.imshow("Final", maskROI)

    key = cv2.waitKey(1)

    if key == 27:
        break

    elif k%256 == 32:
        # SPACE pressed
        img_name = "/Users/cesaralmendarez/Desktop/DeepASL/test_images/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, maskROI)
        print("{} written!".format(img_name))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
