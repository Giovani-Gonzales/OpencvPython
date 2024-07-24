import cv2
import numpy as np
import pytesseract

# Configuração do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

def buscaRetanguloPlaca(source, cascade_path):
    video = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    plate_cascade = cv2.CascadeClassifier(cascade_path)

    cv2.namedWindow('FRAME')
    cv2.setMouseCallback('FRAME', onMouseClick)

    while video.isOpened():
        ret, frame = video.read()

        if not ret or frame is None:
            print("Falha ao capturar frame da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        height, width = frame.shape[:2]
        top_left_x = int(width * 0.25)
        bottom_right_x = int(width * 0.75)
        top_left_y = int(height * 0.7)

        area = frame[top_left_y:height, top_left_x:bottom_right_x]

        if area.size == 0:
            print("Área recortada está vazia.")
            continue

        # Processamento da área para melhorar visualização
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        img_result = cv2.convertScaleAbs(img_result, alpha=1.5, beta=0)
        img_result = cv2.adaptiveThreshold(img_result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)
        

        # Aplica suavização gaussiana
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        # Exibe a área processada na janela 'FRAME'
        cv2.imshow('FRAME', img_result)

        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cv2.line(frame, (0, top_left_y), (width, top_left_y), (0, 0, 255), 1)
        cv2.line(frame, (top_left_x, 0), (top_left_x, height), (0, 0, 255), 1)
        cv2.line(frame, (bottom_right_x, 0), (bottom_right_x, height), (0, 0, 255), 1)

        desenhaContornoMaiorArea(contornos, area)

        cv2.imshow('RES', frame)  # Exibe a imagem original com o contorno desenhado

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

    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Preenchimento dos caracteres
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_fill)
    
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
    source = 1  # Usar a câmera padrão
    cascade_path = 'C:\\Users\\admin\\Documents\\GitHub\\OpencvPython\\cascades\\haarcascade_license_plate_rus_16stages.xml'

    buscaRetanguloPlaca(source, cascade_path)
    reconhecimentoOCR()