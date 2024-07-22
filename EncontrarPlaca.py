import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Variável global para armazenar se o clique ocorreu
clicked = False

def desenhaContornoMaiorArea(contornos, imagem):
    maior_area = 0
    melhor_contorno = None

    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area:
            maior_area = area
            melhor_contorno = c
    
    if melhor_contorno is not None:
        perimetro = cv2.arcLength(melhor_contorno, True)
        approx = cv2.approxPolyDP(melhor_contorno, 0.03 * perimetro, True)
        if len(approx) == 4:
            (x, y, lar, alt) = cv2.boundingRect(melhor_contorno)
            cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 3)
            roi = imagem[y:y + alt, x:x + lar]
            cv2.imwrite("output/roi.png", roi)

def onMouseClick(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True

def buscaRetanguloPlaca(source):
    video = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    cv2.namedWindow('FRAME')
    cv2.setMouseCallback('FRAME', onMouseClick)

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

        desenhaContornoMaiorArea(contornos, area)

        cv2.imshow('RES', area)

        # Verifica se ocorreu um clique na tela
        if clicked:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    preProcessamentoRoi()  # Chamada para processar a ROI após interromper a captura
    cv2.destroyAllWindows()

def preProcessamentoRoi():
    # Carrega a imagem da ROI
    img_roi = cv2.imread("output/roi.png")
    if img_roi is None:
        print("Erro ao carregar imagem da ROI.")
        return

    # Redimensiona a imagem
    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica limiarização
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

    # Aplica desfoque gaussiano
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Salva a imagem processada para OCR
    cv2.imwrite("output/roi-ocr.png", img)

    return img

def reconhecimentoOCR():
    # Carrega a imagem processada para OCR
    img_roi_ocr = cv2.imread("output/roi-ocr.png")
    if img_roi_ocr is None:
        print("Erro ao carregar imagem para OCR.")
        return

    # Configuração para o Tesseract
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    
    # Realiza OCR na imagem e exibe o resultado
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)
    
    # Limita o resultado para 7 caracteres
    resultado_limitado = saida[:7]
    
    print(resultado_limitado)
    return resultado_limitado

if __name__ == "__main__":
    source = 0  # Usar a câmera padrão

    buscaRetanguloPlaca(source)
    reconhecimentoOCR()
