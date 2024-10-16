import cv2
import numpy as np

elipse = cv2.imread('imgs/elipse.png', cv2.IMREAD_GRAYSCALE)
if elipse is not None:
    _, elipse_binaria = cv2.threshold(elipse, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('imgs/binarias/elipse_binaria.png', elipse_binaria)

hexagono = cv2.imread('imgs/hexagono.png', cv2.IMREAD_GRAYSCALE)
if hexagono is not None:
    _, hexagono_binaria = cv2.threshold(hexagono, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('imgs/binarias/hexagono_binaria.png', hexagono_binaria)

triangulo = cv2.imread('imgs/triangulo.png', cv2.IMREAD_GRAYSCALE)
if triangulo is not None:
    _, triangulo_binaria = cv2.threshold(triangulo, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('imgs/binarias/triangulo_binaria.png', triangulo_binaria)

not_elipse_binaria = cv2.bitwise_not(elipse_binaria)
cv2.imwrite('imgs/op-logicas/not_elipse_binaria.png', not_elipse_binaria)

hexagono_and_triangulo_binaria = cv2.bitwise_and(hexagono_binaria, triangulo_binaria)
cv2.imwrite('imgs/op-logicas/hexagono_and_triangulo_binaria.png', hexagono_and_triangulo_binaria)

hexagono_or_triangulo_binaria = cv2.bitwise_or(hexagono_binaria, triangulo_binaria)
cv2.imwrite('imgs/op-logicas/hexagono_or_triangulo_binaria.png', hexagono_or_triangulo_binaria)

# todas las imagenes tienen las mismas dimensiones.
filas, columnas = elipse_binaria.shape

# se hace una traslación hacia la derecha y hacia abajo.
tx, ty = 100, 75
matriz_traslacion = np.float32(
    [
        [1, 0, tx],
        [0, 1, ty]
    ]
)

elipse_binaria_transladada = cv2.warpAffine(elipse_binaria, matriz_traslacion, (columnas, filas))
cv2.imwrite("imgs/transf-geom/elipse_binaria_trasladada.png", elipse_binaria_transladada)

centro = (columnas // 2, filas // 2)
theta = 270
escala = 1

matriz_rotacion = cv2.getRotationMatrix2D(centro, theta, escala)
hexagono_binaria_rotada = cv2.warpAffine(hexagono_binaria, matriz_rotacion, (columnas, filas))
cv2.imwrite("imgs/transf-geom/hexagono_binaria_rotada.png", hexagono_binaria_rotada)

triangulo_binaria_escalada = cv2.resize(triangulo_binaria, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_LINEAR)
cv2.imwrite("imgs/transf-geom/triangulo_binaria_escalada.png", triangulo_binaria_escalada)

racoon1 = cv2.imread("imgs/racoon1.jpg", cv2.IMREAD_GRAYSCALE)
racoon2 = cv2.imread("imgs/racoon2.jpg", cv2.IMREAD_GRAYSCALE)

racoons_and = cv2.bitwise_and(racoon1, racoon2)
cv2.imwrite("imgs/racoons_and.jpg", racoons_and)

racoons_or = cv2.bitwise_or(racoon1, racoon2)
cv2.imwrite("imgs/racoons_or.jpg", racoons_or)

racoons_xor = cv2.bitwise_xor(racoon1, racoon2)
cv2.imwrite("imgs/racoons_xor.jpg", racoons_xor)

racoon1_not = cv2.bitwise_not(racoon1)
cv2.imwrite("imgs/racoon1_not.jpg", racoon1_not)

racoon2_not = cv2.bitwise_not(racoon2)
cv2.imwrite("imgs/racoon2_not.jpg", racoon2_not)
