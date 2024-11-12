# Descargue cuatro imágenes: alto y bajo contraste y alta y baja iluminación.
# Implemente las técnicas basadas en el procesamiento de histograma: 
#   ecualización de histograma global y local, cálculo de media y varianza global y local.
# Prueba las técnicas basadas en el procesamiento de histograma sobre las imágenes descargadas en el punto 1.
#   De acuerdo a la media y varianza (global y local) deberá recomendar y aplicar un procesamiento local o global de cada imagen.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def read_images(image_paths: dict) -> list[np.array]:
    images = []

    for image_path in image_paths.values():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: 
            print(f"Image not found on path: {image_path}")
            exit(1)

        images.append(image)
    
    return images


def read_image(image_path: str) -> np.array:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Image not found on path: {image_path}")
        exit(1)
    
    return image

# TODO: esto
def write_images(images: list[np.array], result_image_paths: list[str]) -> None:
    pass

# TODO: dar nombre a las imagenes; la original y el resultado (metodo usado).
def show_images(original, result) -> None:
    plt.subplot(1, 2, 1)
    # plt.title("Imagen original (alto contraste).")
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # plt.title("Imagen ecualización de histograma global (alto contraste).")
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()


def equalize_hist(image: np.array) -> np.array:
    return cv2.equalizeHist(image)


def equalize_hist_local(image: np.array) -> np.array:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


if __name__ == "__main__":
    image_paths = {
        '1': "imgs/alto_contraste.jpg",
        '2': "imgs/bajo_contraste.jpg",
        '3': "imgs/alta_iluminacion.jpg",
        '4': "imgs/baja_iluminacion.jpg"
    }
    img_ac, img_bc, img_ai, img_bi = read_images(image_paths)

    # TODO: dependiendo si el directorio existe, abrir, procesar y guardar TODAS las imagenes.
    # las imagenes resultado se guardan en un subdirectorio en esta misma carpeta. 
    current_path = os.path.dirname(os.path.abspath(__file__))
    img_result_directory = "img-resultados" 
    img_result_path = os.path.join(current_path, img_result_directory)
    os.makedirs(img_result_path, exist_ok=True)
    
    # TODO: mejorar esta parte, hacerlo en una funcion o yo qué sé.
    img_ac_global_hist = equalize_hist(img_ac)
    cv2.imwrite(img_result_directory + '/ac_histograma_global.jpeg', img_ac_global_hist)

    img_bc_global_hist = equalize_hist(img_bc)
    cv2.imwrite(img_result_directory + '/bc_histograma_global.jpeg', img_bc_global_hist)

    img_ai_global_hist = equalize_hist(img_ai)
    cv2.imwrite(img_result_directory + '/ai_histograma_global.jpeg', img_ai_global_hist)

    img_bi_global_hist = equalize_hist(img_bi)
    cv2.imwrite(img_result_directory + '/bi_histograma_global.jpeg', img_bi_global_hist)


    while True:
        print("\nOpciones (imágenes): ")
        print("  [1]: Alto contraste.")
        print("  [2]: Bajo contraste.")
        print("  [3]: Alta iluminación.")
        print("  [4]: Baja iluminación")
        print("  [5]: Salir.")

        image_option = input("Elija un numero (1-5): ")
        if image_option in image_paths: image = read_image(image_paths[image_option])
        elif image_option == "5": break

        print("\nOpciones (ecualización de histograma): ")
        print("  [1]: Local.")
        print("  [2]: Global.")
        print("  [3]: Cancelar.")

        equalization_option = input("Elija un numero (1-3): ")
        if equalization_option == '1':
            result = equalize_hist(image)
            show_images(image, result)
        elif equalization_option == '2': 
            result = equalize_hist_local(image)
            show_images(image, result)
        elif equalization_option == '3': continue
        