import cv2 as cv
import numpy as np

# Lê a imagem em BGR
img = cv.imread('./train/S/1009.png', cv.IMREAD_GRAYSCALE)

inv = cv.bitwise_not(img)

ret, thresh = cv.threshold(inv, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

cv.imwrite('./output.png', opening)
