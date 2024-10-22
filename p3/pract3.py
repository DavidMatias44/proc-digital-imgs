# Utilizando una herramienta de PDI, realice lo siguiente:
#   1. Descargue tres imágenes con: alto y bajo contraste, y poca iluminación.
#   2. Implemente las técnicas de transformación de intensidad básicas:
#      negativo, transf. logarítmica, transf. gamma, estiramiento de contraste, 
#      rebanada de nivel de intensidad y rebanada de plano de bit.
#   3. Pruebe las técnicas de transformación de intensidad básicas sobre las
#      imágenes descargadas en el punto 1.

import cv2
import numpy as np

img_ac = cv2.imread('imgs/alto_contraste.jpeg', cv2.IMREAD_GRAYSCALE)
if img_ac is None:
    print("Image not found.")
    exit(1)
img_ac_bits = np.array(img_ac)

img_bc = cv2.imread('imgs/bajo_contraste.jpeg', cv2.IMREAD_GRAYSCALE)
if img_bc is None:
    print("Image not found.")
    exit(1)
img_bc_bits = np.array(img_bc)

img_pi = cv2.imread('imgs/poca_ilum.jpg', cv2.IMREAD_GRAYSCALE)
if img_pi is None:
    print("Image not found.")
    exit(1)
img_pi_bits = np.array(img_pi)


def negativo(img: np.array) -> np.array:
    return cv2.bitwise_not(img)

def transf_log(img: np.array) -> np.array:
    img_float = np.float32(img) / 255.0
    # scale_constant = 255 / np.log(1 + np.max(img_float))
    img_tlog = 3 * np.log(1 + img_float)

    return np.uint8(np.clip(img_tlog * 255, 0, 255))


if __name__ == "__main__":
    cv2.imwrite("imgs/negs/ac_neg.jpg", negativo(img_ac))
    cv2.imwrite("imgs/negs/bc_neg.jpg", negativo(img_bc))
    cv2.imwrite("imgs/negs/pi_neg.jpg", negativo(img_pi))

    cv2.imwrite("imgs/transf-logs/ac_tlog.jpg", transf_log(img_ac))
    cv2.imwrite("imgs/transf-logs/bc_tlog.jpg", transf_log(img_bc))
    cv2.imwrite("imgs/transf-logs/pi_tlog.jpg", transf_log(img_pi))