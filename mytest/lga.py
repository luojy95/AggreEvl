import cv2
import numpy as np
import matplotlib.pyplot as plt

gocator_img_name = "outputs/processed_gocator.png"
linescan_img_name = "outputs/processed_linescan.png"


img1 = cv2.imread(gocator_img_name, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(linescan_img_name, cv2.IMREAD_GRAYSCALE)

concat = np.concatenate([img1, img2], axis=0)

plt.imshow(concat, cmap="gray")
plt.show()
