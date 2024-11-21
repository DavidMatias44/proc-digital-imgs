# Utilizando una herramienta de PDI, realice los siguiente.
# 1. Descargue cuatro imágenes: alto y bajo contraste, y alta y baja iluminación.
# 2. Implemente los filtros suavizantes y realzantes: promedio, mediana, máximo y mínimo,
#    Laplaciano y gradiente. Deberá utilizar distintos tamaños de máscaras (image: np.array,3x3, 5x5, etc.),
#    excepto para el Laplaciano y para el gradiente.
# 3. Pruebe los filtros suavizantes y realzantes sobre las imágenes descargadas en el punto 1.
#    Deberá recomendar un filtro para cada imagen.


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


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


def add_zero_padding(image: np.array, padding: np.uint8) -> np.array:
    height, width = image.shape

    padded_image = np.zeros((height + 2 * padding, width + 2 * padding))
    padded_image[padding:padding + height, padding:padding + width] = image

    return padded_image


def filtro_promedio(image: np.array, kernel_size: np.uint8):
    padding = kernel_size // 2
    result = np.zeros_like(image)
    padded_image = add_zero_padding(image, kernel_size // 2)

    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.mean(vecinity)

    return result


def filtro_mediana(image: np.array, kernel_size: np.uint8):
    padding = kernel_size // 2
    result = np.zeros_like(image)
    padded_image = add_zero_padding(image, kernel_size // 2)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.median(vecinity)

    return result


def filtro_maximo(image: np.array, kernel_size: np.uint8):
    padding = kernel_size // 2
    result = np.zeros_like(image)
    padded_image = add_zero_padding(image, kernel_size // 2)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.max(vecinity)

    return result


def filtro_minimo(image: np.array, kernel_size: np.uint8):
    padding = kernel_size // 2
    result = np.zeros_like(image)
    padded_image = add_zero_padding(image, padding)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.min(vecinity)

    return result


def filtro_laplaciano(image: np.array):
    matriz = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ])
    result = np.zeros_like(image)
    temp = np.zeros_like(image)
    padded_image = add_zero_padding(image, 1)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + 3, j:j + 3]
            temp[i, j] = np.sum(matriz * vecinity)

    # el centro de la matriz utilizada es negativo, por ello la cte es negativa, como la cte = 1, entonces simplemente se resta.
    result = image - temp

    return np.clip(result, 0, 255)


def filtro_gradiente(image: np.array):
    grad_x = np.array([
        [-1,  -2, -1],
        [ 0,   0,  0],
        [ 1,   2,  1],
    ])
    grad_y = np.array([
        [-1, 0, -1],
        [-2, 0, -2],
        [-1, 0, -1],
    ])
    result = np.zeros_like(image)
    padded_image = add_zero_padding(image, 1)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            vecinity = padded_image[i:i + 3, j:j + 3]
            result[i, j] = np.sum([np.abs(grad_x * vecinity), np.abs(grad_y * vecinity)])

    return np.clip(result, 0, 255)


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
        print("  [1]: Filtro promedio.")
        print("  [2]: Filtro mediana.")
        print("  [3]: Filtro maximo.")
        print("  [4]: Filtro minimo.")
        print("  [5]: Filtro Laplaciano.")
        print("  [6]: Filtro Gradiente.")
        print("  [7]: Cancelar.")

        filter_option = input("Elija un numero (1-7): ")
        if filter_option < '5':
            kernel_size = int(input("\nIntroduzca el tamano de la ventana: "))

            if filter_option == '1':
                result = filtro_promedio(image, kernel_size)

                show_images(image, result, aux2[image_option], "filtro promedio")
                cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-prom.jpg", result)
            elif filter_option == '2':
                result = filtro_mediana(image, kernel_size)

                show_images(image, result, aux2[image_option], "filtro mediana")
                cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-med.jpg", result)
            elif filter_option == '3':
                result = filtro_maximo(image, kernel_size)

                show_images(image, result, aux2[image_option], "filtro maximo")
                cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-max.jpg", result)
            elif filter_option == '4':
                result = filtro_minimo(image, kernel_size)

                show_images(image, result, aux2[image_option], "filtro minimo")
                cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-min.jpg", result)
        elif filter_option == '5':
            result = filtro_laplaciano(image)

            show_images(image, result, aux2[image_option], "filtro laplaciano")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-lapl.jpg", result)
        elif filter_option == '6':
            result = filtro_gradiente(image)

            show_images(image, result, aux2[image_option], "filtro gradiente")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-grad.jpg", result)
        else: continue

        #     result = local_eq_hist(image, window_filter_size)
        #     show_images(image, result, aux2[image_option], "ecualizacion de histograma local")
        #     cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-local.jpeg", result)
        # elif equalization_option == '2': 
        #     result = global_eq_hist(image)
        #     show_images(image, result, aux2[image_option], "ecualizacion de histograma global")
            # cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-global.jpeg", result)
        # elif equalization_option == '3': 
            # window_filter_size = int(input("\nIntroduzca el tamano de la ventana: "))

            # result = realce_usando_estadistica_del_histograma(image, window_filter_size)
            # show_images(image, result, aux2[image_option], "estadistica del histograma")
            # cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-uso-est-hist.jpeg", result)
