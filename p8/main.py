# Descargue tres imágenes en tonos de gris de escenas: deportiva, edificio y gente.
# Implemente las operaciones de detección de líneas, detección de ejes (Sobel, Marr-Mildreth, Canny) y segmentación por umbralización global 
# (básica y por Otsu).
# Prueba las operaciones y algoritmos sobre las imágenes descargadas en el punto 1. Deberá recomendar un operador o algoritmo para cada imagen.

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


def deteccion_lineas(img):
    pass


def deteccion_ejes_sobel(img, ksize=3):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_magnitud = cv2.magnitude(grad_x, grad_y)
    sobel_bordes = np.uint8(np.clip(grad_magnitud, 0, 255))
    return sobel_bordes


def deteccion_ejes_marr_hildret(img, sigma=0.125):
    suavizado = cv2.GaussianBlur(img, (0, 0), sigma)
    laplaciano = cv2.Laplacian(suavizado, cv2.CV_64F)
    marr_hildreth = np.uint8(np.clip(np.absolute(laplaciano), 0, 255))

    return marr_hildreth


def deteccion_ejes_canny(img, umbral1=50, umbral2=150):
    bordes_canny = cv2.Canny(img, umbral1, umbral2)
    return bordes_canny


def umbralizacion_global_basica(img):
    pass


def umbralizacion_global_otsu(img):
    pass


if __name__ == "__main__":
    img = read_image('imgs/deportiva.jpg')
    cv2.imwrite("imgs/deportiva1.jpg", img)
    image_paths = {
        '1': "imgs/deportiva.jpg",
        '2': "imgs/edificio.jpg",
        '3': "imgs/gente.jpg"
    }
    aux = {"1": "deportiva", "2": "edificios", "3": "gente"}
    img_result_directory = "img-resultados" 

    current_path = os.path.dirname(os.path.abspath(__file__))
    img_result_path = os.path.join(current_path, img_result_directory)
    os.makedirs(img_result_path, exist_ok=True)
    
    while True:
        print("\nOpciones (imágenes): ")
        print("  [1]: Escena deportiva.")
        print("  [2]: Escena edificios.")
        print("  [3]: Escena gente.")
        print("  [4]: Salir.")

        image_option = input("Elija un numero (1-4): ")
        if image_option in image_paths: image = read_image(image_paths[image_option])
        elif image_option == "4": break
        else: continue

        print("\nOpciones: ")
        print("  [1]: Detección de líneas.")
        print("  [2]: Detección de ejes (Sobel).")
        print("  [3]: Detección de ejes (Marr-Mildreth).") 
        print("  [4]: Detección de ejes (Canny).")
        print("  [5]: Segementación por umbralización global (básica)")
        print("  [6]: Segementación por umbralización global (Otsu)")
        print("  [7]: Cancelar.")

        option = input("Elija un numero (1-7): ")
        if option == '1':
            pass
        elif option == '2':
            ksize = int(input("Introduzca el tamaño del kernel: "))
            result = deteccion_ejes_sobel(image, ksize)
            show_images(image, result, aux[image_option], "Sobel")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-sobel.jpg", result)
        elif option == '3':
            result = deteccion_ejes_marr_hildret(image)
            show_images(image, result, aux[image_option], "Marr-Hildreth")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-marr-hildreth.jpg", result)
        elif option == '4':
            umbral1 = int(input("Introduzca el umbral menor: "))
            umbral2 = int(input("Introduzca el umbral mayor: "))

            result = deteccion_ejes_canny(image, umbral1, umbral2)
            show_images(image, result, aux[image_option], "Canny")
            cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-canny.jpg", result)

#         if option < '5':
#             kernel_size = int(input("Introduzca el tamano del kernel: "))
#             # elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#             # elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
#             elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            

#             if option == '1':
#                 result = erosion(image, elemento_estructurante)

#                 show_images(image, result, aux[image_option], "erosion")
#                 cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-erosion.jpg", result)
#             elif option == '2':
#                 result = dilatacion(image, elemento_estructurante)

#                 show_images(image, result, aux[image_option], "dilatacion")
#                 cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-dilatacion.jpg", result)
#             elif option == '3':
#                 result = apertura(image, elemento_estructurante)

#                 show_images(image, result, aux[image_option], "apertura")
#                 cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-apertura.jpg", result)
#             elif option == '4':
#                 result = cierre(image, elemento_estructurante)

#                 show_images(image, result, aux[image_option], "cierre")
#                 cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-cierre.jpg", result)
#         elif option == '5':
#             result = extraccion_limites(image)

#             show_images(image, result, aux[image_option], "extraccion limites")
#             cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-ext_limites.jpg", result)
#         elif option == '6':
#             result = rellenado_hoyos(image)
#             result = apertura(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

#             show_images(image, result, aux[image_option], "rellenado hoyos")
#             cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-rell_hoyos.jpg", result)
#         elif option == '7':
#             num_etiquetas, etiquetas = extraer_componentes_conectados(image)

#             print(f"Numero de componentes hallados: {num_etiquetas}") 
#             # show_images(image, etiquetas, aux[image_option], "extraccion componentes conectados")
#             # cv2.imwrite(f"{img_result_directory}/{aux[image_option]}-ext_comp_conectados.jpg", etiquetas)
#         else: continue