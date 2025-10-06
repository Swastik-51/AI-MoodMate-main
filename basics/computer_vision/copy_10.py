#CODE 10

import cv2

img = cv2.imread("resources/man.jpg", 0)  # Using sample image from resources folder

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()