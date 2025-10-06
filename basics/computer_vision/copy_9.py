#CODE 9

import cv2

img = cv2.imread("resources/man.jpg", 0)  # Load sample image in grayscale

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("Thresholded", thresh)
cv2.destroyAllWindows()