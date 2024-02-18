import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('car.jpeg')
blur = cv.bilateralFilter(img,45,200,200)

plt. subplot(121),plt.imshow(img),plt. title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]),plt.yticks([])

plt.show()
