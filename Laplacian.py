import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)
ret, original_img = cap.read()
cap.release()

gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)

laplacian = cv.Laplacian(blur_img, cv.CV_64F)
laplacian_abs = cv.convertScaleAbs(laplacian)

plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis("off")

plt.subplot(122)
plt.imshow(laplacian_abs, cmap='gray')
plt.title('Laplacian')
plt.axis("off")

plt.show()
