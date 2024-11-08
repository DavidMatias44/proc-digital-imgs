# Descargue cuatro imágenes: alto y bajo contraste y alta y baja iluminación.
# Implemente las técnicas basadas en el procesamiento de histograma: 
#   ecualización de histograma global y local, cálculo de media y varianza global y local.
# Prueba las técnicas basadas en el procesamiento de histograma sobre las imágenes descargadas en el punto 1.
#   De acuerdo a la media y varianza (global y local) deberá recomendar y aplicar un procesamiento local o global de cada imagen.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def read_images(image_paths: list[str]) -> list[np.array]:
    images = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: 
            print(f"Image not found on path: {image_path}")
            exit(1)

        images.append(image)
    
    return images


def equalize_hist(image: np.array) -> np.array:
    return cv2.equalizeHist(image)


if __name__ == "__main__":
    image_paths = [
        "imgs/alto_contraste.jpg",
        "imgs/bajo_contraste.jpg",
        "imgs/alta_iluminacion.jpg",
        "imgs/baja_iluminacion.jpg"
    ]
    img_ac, img_bc, img_ai, img_bi = read_images(image_paths)

    # las imagenes resultado se guardan en un subdirectorio en esta misma carpeta. 
    current_path = os.path.dirname(os.path.abspath(__file__))
    img_result_directory = "img-resultados" 
    img_result_path = os.path.join(current_path, img_result_directory)
    os.makedirs(img_result_path, exist_ok=True)
    
    img_ac_global_hist = equalize_hist(img_ac)
    cv2.imwrite(img_result_directory + '/ac_histograma_global.jpeg', img_ac_global_hist)

    img_bc_global_hist = equalize_hist(img_bc)
    cv2.imwrite(img_result_directory + '/bc_histograma_global.jpeg', img_bc_global_hist)

    img_ai_global_hist = equalize_hist(img_ai)
    cv2.imwrite(img_result_directory + '/ai_histograma_global.jpeg', img_ai_global_hist)

    img_bi_global_hist = equalize_hist(img_bi)
    cv2.imwrite(img_result_directory + '/bi_histograma_global.jpeg', img_bi_global_hist)

    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.title("Imagen original (alto contraste).")
    plt.imshow(img_ac, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Imagen ecualización de histograma global (alto contraste).")
    plt.imshow(img_ac_global_hist, cmap='gray')
    plt.axis('off')

    plt.show()