import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

ret, original_img = cap.read()
cap.release()

blur_img = cv.GaussianBlur(original_img, (5, 5), 0)
edges = cv.Canny(blur_img, 50, 150)

plt.figure(figsize=(20,20))

plt.subplot(221)
plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis("off")

plt.subplot(222)
plt.imshow(edges, cmap='gray')
plt.title('Canny')
plt.axis("off")

plt.show()
