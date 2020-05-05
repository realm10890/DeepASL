import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

l = cv2.imread("opencv_frame_37_28x28.png")
l2 = cv2.imread("opencv_frame_37_28x28.png")

grayl = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
grayl2 = cv2.cvtColor(l2, cv2.COLOR_BGR2GRAY)

both = [grayl, grayl2]

bothNumpy = np.array(both)

print(bothNumpy)

cv2.imshow("L", l)
cv2.imshow("grayL", grayl)
cv2.waitKey(0)
cv2.destroyAllWindows()
