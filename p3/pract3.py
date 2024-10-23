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

img_bc = cv2.imread('imgs/bajo_contraste.jpeg', cv2.IMREAD_GRAYSCALE)
if img_bc is None:
    print("Image not found.")
    exit(1)

img_pi = cv2.imread('imgs/poca_ilum.jpg', cv2.IMREAD_GRAYSCALE)
if img_pi is None:
    print("Image not found.")
    exit(1)


def negativo(img: np.array) -> np.array:
    return cv2.bitwise_not(img)


def transf_log(img: np.array) -> np.array:
    img_float = np.float32(img) / 255.0
    # scale_constant = 255 / np.log(1 + np.max(img_float))
    img_tlog = 3 * np.log(1 + img_float)

    return np.uint8(np.clip(img_tlog * 255, 0, 255))


def transf_gamma(img: np.array, gamma: float) -> np.array:
    img_float = np.float32(img) / 255.0
    img_tgamma = np.power(img_float, gamma)
    
    return np.uint8(np.clip(img_tgamma * 255, 0, 255))


def estiramiento_contraste(img: np.array) -> np.array:
    valor_minimo, valor_maximo = np.min(img), np.max(img)
    img_estcont = (img - valor_minimo) * (255 / (valor_maximo - valor_minimo))
    
    return np.uint8(img_estcont)


def rebanada_nvls_intensidad(img: np.array, rango_min: np.uint8, rango_max: np.uint8, valor_especif: np.uint8, mantener_otro_valores: bool) -> np.array:
    img_cp = np.zeros_like(img)

    if mantener_otro_valores:
        img_cp = np.copy(img)

    mascara = (img >= rango_min) & (img <= rango_max)
    img_cp[mascara] = valor_especif

    return img_cp

def rebanada_plano_bits(img: np.array, num_plano: np.uint8) -> np.array:
    return np.uint8((img >> num_plano) & 1) * 255


if __name__ == "__main__":
    cv2.imwrite("imgs/negs/ac_neg.jpg", negativo(img_ac))
    cv2.imwrite("imgs/negs/bc_neg.jpg", negativo(img_bc))
    cv2.imwrite("imgs/negs/pi_neg.jpg", negativo(img_pi))

    cv2.imwrite("imgs/transf-logs/ac_tlog.jpg", transf_log(img_ac))
    cv2.imwrite("imgs/transf-logs/bc_tlog.jpg", transf_log(img_bc))
    cv2.imwrite("imgs/transf-logs/pi_tlog.jpg", transf_log(img_pi))

    gamma_value = 0.5
    cv2.imwrite("imgs/transf-gamma/ac_tgamma.jpg", transf_gamma(img_ac, gamma_value))
    cv2.imwrite("imgs/transf-gamma/bc_tgamma.jpg", transf_gamma(img_bc, gamma_value))
    cv2.imwrite("imgs/transf-gamma/pi_tgamma.jpg", transf_gamma(img_pi, gamma_value))

    cv2.imwrite("imgs/est-contraste/ac_estcont.jpg", estiramiento_contraste(img_ac))
    cv2.imwrite("imgs/est-contraste/bc_estcont.jpg", estiramiento_contraste(img_bc))
    cv2.imwrite("imgs/est-contraste/pi_estcont.jpg", estiramiento_contraste(img_pi))

    rango_min = 127
    rango_max = 191
    valor_especif = 255
    cv2.imwrite("imgs/rebanada-nvl-intensidad/ac_rebnvlint.jpg", rebanada_nvls_intensidad(img_ac, rango_min, rango_max, valor_especif, False))
    cv2.imwrite("imgs/rebanada-nvl-intensidad/bc_rebnvlint.jpg", rebanada_nvls_intensidad(img_bc, rango_min, rango_max, valor_especif, False))
    cv2.imwrite("imgs/rebanada-nvl-intensidad/pi_rebnvlint.jpg", rebanada_nvls_intensidad(img_pi, rango_min, rango_max, valor_especif, False))

    cv2.imwrite("imgs/rebanada-nvl-intensidad/ac_rebnvlintv2.jpg", rebanada_nvls_intensidad(img_ac, rango_min, rango_max, valor_especif, True))
    cv2.imwrite("imgs/rebanada-nvl-intensidad/bc_rebnvlintv2.jpg", rebanada_nvls_intensidad(img_bc, rango_min, rango_max, valor_especif, True))
    cv2.imwrite("imgs/rebanada-nvl-intensidad/pi_rebnvlintv2.jpg", rebanada_nvls_intensidad(img_pi, rango_min, rango_max, valor_especif, True))

    plano_bits_img_ac = [rebanada_plano_bits(img_ac, num_plano) for num_plano in range(8)]
    plano_bits_img_bc = [rebanada_plano_bits(img_bc, num_plano) for num_plano in range(8)]
    plano_bits_img_pi = [rebanada_plano_bits(img_pi, num_plano) for num_plano in range(8)]

    contador = 0
    for plano in plano_bits_img_ac:
        cv2.imwrite(f"imgs/rebanada-plano-bits/alto-contraste/img_ac_plano{contador}.jpg", plano)
        contador += 1

    contador = 0
    for plano in plano_bits_img_bc:
        cv2.imwrite(f"imgs/rebanada-plano-bits/bajo-contraste/img_bc_plano{contador}.jpg", plano)
        contador += 1

    contador = 0
    for plano in plano_bits_img_pi:
        cv2.imwrite(f"imgs/rebanada-plano-bits/poca-iluminacion/img_pi_plano{contador}.jpg", plano)
        contador += 1
