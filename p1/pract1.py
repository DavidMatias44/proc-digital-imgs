import cv2

imagen = cv2.imread('../../p.jpg')

if imagen is None:
    print("No se pudo leer la imagen.")
else:
    cv2.imshow('Imagen', imagen)

    cv2.imwrite('./s.bmp', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()