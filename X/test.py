import numpy as np
import cv2 as cv
img = cv.imread('X/image23.jpg')
output = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (9, 9), 0)
canny = cv.Canny(blur, 30, 80)
circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 20,
                          param1=50, param2=30, minRadius=20, maxRadius=0)
detected_circles = np.uint16(np.around(circles))
for (x, y ,r) in detected_circles[0, :]:
    cv.circle(output, (x, y), r, (0, 0, 0), 3)
    cv.circle(output, (x, y), 2, (0, 255, 255), 3)


cv.imshow('canny',canny)
cv.waitKey(0)
cv.destroyAllWindows()