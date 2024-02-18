import cv2
from matplotlib import pyplot as plt

img = cv2.imread("cat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blur_averaging = cv2.blur(img_rgb, (35, 35))
blur_gaussian = cv2.GaussianBlur(img_rgb, (35, 35), 0)
blur_median = cv2.medianBlur(img_rgb, 35)
blur_bilateral = cv2.bilateralFilter(img_rgb, 65, 145, 145)

fig, ax = plt.subplots(3, 3, figsize=(9, 9))

ax[2, 1].remove()
ax[0, 1].remove()
ax[1, 0].remove()
ax[1, 2].remove()

ax_center = fig.add_subplot(3, 3, 5)
ax_center.imshow(img_rgb)
ax_center.set_title('Original', color='black', fontsize=11,fontweight='bold', y=0.87, x=0.55)
ax_center.axis('off')

ax_averaging = fig.add_subplot(3, 3, 1)
ax_averaging.imshow(blur_averaging)
ax_averaging.set_title('Averaging', color='black', fontsize=11,fontweight='bold', y=0.87, x=0.55)
ax_averaging.axis('off')

ax_gaussian = fig.add_subplot(3, 3, 3)
ax_gaussian.imshow(blur_gaussian)
ax_gaussian.set_title('Gaussian', color='black', fontsize=11,fontweight='bold', y=0.87, x=0.55)
ax_gaussian.axis('off')

ax_median = fig.add_subplot(3, 3, 7)
ax_median.imshow(blur_median)
ax_median.set_title('Median', color='black', fontsize=11,fontweight='bold', y=0.87, x=0.55)
ax_median.axis('off')

ax_bilateral = fig.add_subplot(3, 3, 9)
ax_bilateral.imshow(blur_bilateral)
ax_bilateral.set_title('Bilateral Filtering', color='black', fontsize=11,fontweight='bold', y=0.87, x=0.55)
ax_bilateral.axis('off')

plt.show()
