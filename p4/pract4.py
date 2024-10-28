import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bc = cv2.imread("imgs/bajo_contraste.jpeg")
if img_bc is None:
    print("Image not found.")
    exit(1)

img_ac = cv2.imread("imgs/alto_contraste.jpeg")
if img_ac is None:
    print("Image not found.")
    exit(1)

img_ai = cv2.imread("imgs/alta_iluminacion.jpeg", cv2.IMREAD_GRAYSCALE)
if img_ai is None:
    print("Image not found.")
    exit(1)

cv2.imwrite("imgs/test.jpeg", img_ai)

imgplt_ai = plt.imshow(img_ai, cmap="gray")
plt.show()
