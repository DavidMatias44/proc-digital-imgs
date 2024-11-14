# Descargue cuatro imágenes: alto y bajo contraste y alta y baja iluminación.
# Implemente las técnicas basadas en el procesamiento de histograma: 
#   ecualización de histograma global y local, cálculo de media y varianza global y local.
# Prueba las técnicas basadas en el procesamiento de histograma sobre las imágenes descargadas en el punto 1.
#   De acuerdo a la media y varianza (global y local) deberá recomendar y aplicar un procesamiento local o global de cada imagen.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

MAX_INTENSITY = 256


def read_image(image_path: str) -> np.array:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Image not found on path: {image_path}")
        exit(1)
    
    return image


def show_images(original: np.array, result: np.array, image_name: str, method: str) -> None:
    plt.subplot(1, 2, 1)
    plt.title(f"Imagen original ({image_name}).")
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Imagen usando: {method} ({image_name}).")
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()
    return


def local_eq_hist(image: np.array, filter_window_size: np.uint16) -> np.array:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(filter_window_size, filter_window_size))
    return clahe.apply(image)


def global_eq_hist(image: np.array) -> np.array:
    histogram = get_histogram(image)
    pdf = get_PDF(histogram)

    result = pdf[image]
    return result 


def get_histogram(image: np.array) -> np.array:
    result = np.zeros(MAX_INTENSITY)    
    image_size = image.shape[0] * image.shape[1]

    for row in image:
        for pixel in row:
            result[pixel] += 1

    return result / image_size


def get_PDF(histogram: np.array) -> np.array:
    result = np.zeros(MAX_INTENSITY)
    histogram = histogram * (MAX_INTENSITY - 1)

    for index in range(MAX_INTENSITY):
        if index == 0:
            result[index] = histogram[index]
        else:
            result[index] = result[index - 1] + histogram[index]

    return np.uint8(np.round(result))


def realce_usando_estadistica_del_histograma(image, window_filter_size):
    result = np.copy(image)
    cte = 0.9
    E = 2.5

    media_global = np.mean(image)
    desv_estandar_global = np.sqrt(np.var(image))

    window_filter = (window_filter_size, window_filter_size)
    media_local = cv2.blur(image, window_filter)
    desv_estandar_local = np.sqrt(cv2.blur(image ** 2, window_filter) - media_local ** 2)

    condicion = media_local <= (cte * media_global)
    print(condicion)
    result[condicion] = np.clip(E * image[condicion], 0, 255)

    return result


if __name__ == "__main__":
    image_paths = {
        '1': "imgs/alto_contraste.jpg",
        '2': "imgs/bajo_contraste.jpg",
        '3': "imgs/alta_iluminacion.jpg",
        '4': "imgs/baja_iluminacion.jpg"
    }
    aux = {"1": "ac", "2": "bc", "3": "ai", "4": "bi"}
    aux2 = {
        "1": "alto contraste",
        "2": "bajo contraste",
        "3": "alta iluminacion",
        "4": "baja iluminacion"
    }
    img_result_directory = "img-resultados" 

    current_path = os.path.dirname(os.path.abspath(__file__))
    img_result_path = os.path.join(current_path, img_result_directory)
    os.makedirs(img_result_path, exist_ok=True)
    
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
        else: continue

        print("\nOpciones: ")
        print("  [1]: Ecualizacion del histograma local.")
        print("  [2]: Ecualizacion del histograma global.")
        print("  [3]: Realce usando estadistica del histograma.")
        print("  [4]: Cancelar.")

        equalization_option = input("Elija un numero (1-4): ")
        if equalization_option == '1':
            window_filter_size = int(input("\nIntroduzca el tamano de la ventana: "))

            result = local_eq_hist(image, window_filter_size)
            show_images(image, result, aux2[image_option], "ecualizacion de histograma local")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-local.jpeg", result)
        elif equalization_option == '2': 
            result = global_eq_hist(image)
            show_images(image, result, aux2[image_option], "ecualizacion de histograma global")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-global.jpeg", result)
        elif equalization_option == '3': 
            window_filter_size = int(input("\nIntroduzca el tamano de la ventana: "))

            result = realce_usando_estadistica_del_histograma(image, window_filter_size)
            show_images(image, result, aux2[image_option], "estadistica del histograma")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-uso-est-hist.jpeg", result)
        else: continue
        