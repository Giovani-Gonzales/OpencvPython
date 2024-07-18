import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\garaujo\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

def desenhaContornos(contornos, imagem):
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(approx) == 4:
                (x, y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                roi = imagem[y:y + alt, x:x + lar]
                cv2.imwrite("output/roi.png", roi)

def buscaRetanguloPlaca(source):
    video = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    while video.isOpened():
        ret, frame = video.read()

        if not ret or frame is None:
            print("Falha ao capturar frame da câmera.")
            break

        height, width = frame.shape[:2]
        top_left_x = int(width * 0.25)
        bottom_right_x = int(width * 0.75)
        top_left_y = int(height * 0.7)

        area = frame[top_left_y:height, top_left_x:bottom_right_x]

        if area.size == 0:
            print("Área recortada está vazia.")
            continue
        
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cv2.line(frame, (0, top_left_y), (width, top_left_y), (0, 0, 255), 1)
        cv2.line(frame, (top_left_x, 0), (top_left_x, height), (0, 0, 255), 1)
        cv2.line(frame, (bottom_right_x, 0), (bottom_right_x, height), (0, 0, 255), 1)

        cv2.imshow('FRAME', frame)

        desenhaContornos(contornos, area)

        cv2.imshow('RES', area)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    preProcessamentoRoi()
    cv2.destroyAllWindows()

def preProcessamentoRoi():
    img_roi = cv2.imread("output/roi.png")
    cv2.imshow("ENTRADA", img_roi)
    if img_roi is None:
        return

    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Escala Cinza", img)

    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Limiar", img)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow("Desfoque", img)

    # Realce de bordas
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow("Bordas", edges)

    # Ajuste de contraste e brilho
    alpha = 1 # Contraste
    beta = 100   # Brilho
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow("Contraste e Brilho", img)

    cv2.imwrite("output/roi-ocr.png", img)

    return img

def reconhecimentoOCR():
    img_roi_ocr = cv2.imread("output/roi-ocr.png")
    if img_roi_ocr is None:
        return

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)

    # Verificar por dois dígitos iguais seguidos
    for i in range(len(saida) - 1):
        if saida[i] == saida[i + 1]:
            print(f"Dois dígitos iguais seguidos encontrados: {saida[i]}{saida[i + 1]}")
            break

    print(saida)
    return saida

if __name__ == "__main__":
    source = 0  # Usar a câmera padrão

    buscaRetanguloPlaca(source)
    preProcessamentoRoi()
    reconhecimentoOCR()
