import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    sobely = cv.Sobel(src=blur_img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    filtered_image_y = cv.convertScaleAbs(sobely)

    laplacian = cv.Laplacian(blur_img, cv.CV_64F)
    filtered_image = cv.convertScaleAbs(laplacian)

    edges = cv.Canny(blur_img, 50, 150)


    #PLOTTING OF VIDS
    plt.clf()

    plt.subplot(221)
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis("off")

    plt.subplot(222)
    plt.imshow(filtered_image_y, cmap='gray')
    plt.title('Sobel y')
    plt.axis("off")

    plt.subplot(223)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Laplacian')
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny')
    plt.axis("off")

    plt.pause(0.01)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
