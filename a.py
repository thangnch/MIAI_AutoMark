import cv2

image = cv2.imread('a.jpg',0)
thres = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,25,10)

cv2.imshow('a',thres)
cv2.waitKey()