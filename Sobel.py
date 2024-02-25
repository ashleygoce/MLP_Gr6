import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

ret, original_img = cap.read()
cap.release()

blur_img = cv.GaussianBlur(original_img, (5, 5), 0)

# Sobel detector
sobelx = cv.Sobel(src=blur_img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
filtered_image_x = cv.convertScaleAbs(sobelx)

sobely = cv.Sobel(src=blur_img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
filtered_image_y = cv.convertScaleAbs(sobely)

sobelxy = cv.Sobel(src=blur_img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
filtered_image_xy = cv.convertScaleAbs(sobelxy)

plt.figure(figsize=(20,20))

plt.subplot(221)
plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis("off")

plt.subplot(222)
plt.imshow(filtered_image_x, cmap='gray')
plt.title('Sobel x')
plt.axis("off")

plt.subplot(223)
plt.imshow(filtered_image_y, cmap='gray')
plt.title('Sobel y')
plt.axis("off")

plt.subplot(224)
plt.imshow(filtered_image_xy, cmap='gray')
plt.title('Sobel xy')
plt.axis("off")

plt.show()
