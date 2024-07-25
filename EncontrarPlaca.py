import cv2
import pytesseract
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import Label
import pandas as pd
import difflib

# Carregar os dados
df = pd.read_csv('dados.csv')

# Configuração do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class WebcamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.lista_placas = list(df['placa'])

        self.title("Webcam App")
        self.geometry("1000x800")

        self.video_label = Label(self)
        self.video_label.pack(padx=10, pady=10)

        self.resultado_label = ctk.CTkLabel(self, text="", font=("Arial", 20))
        self.resultado_label.pack(pady=20)

        self.btn = ctk.CTkButton(self, text="Concluir", command=self.concluir)
        self.btn.pack(pady=10, padx=10)

        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Erro ao abrir a câmera.")
            return

        self.update_video()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def desenhaContornoMaiorArea(self, contornos, imagem):
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

    def preProcessamentoRoi(self):
        # Carrega a imagem da ROI
        img_roi = cv2.imread("output/roi.png")
        if img_roi is None:
            print("Erro ao carregar imagem da ROI.")
            return None

        img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
        # Converte para escala de cinza
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplica o limiar binário com um valor fixo
        _, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
        
        # Aplica suavização gaussiana
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Ajusta o brilho e o contraste
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    
        # Define o tamanho do elemento estruturante para afinação
        kernel_size = (3, 3)  # Tamanho maior para afinar mais os caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # Aplica a operação de abertura morfológica para afinar as letras
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
        # Salva a imagem processada para OCR
        cv2.imwrite("output/roi-ocr.png", img)

        return self.reconhecimentoOCR()

    def reconhecimentoOCR(self):
        # Carrega a imagem processada para OCR
        img_roi_ocr = cv2.imread("output/roi-ocr.png")
        if img_roi_ocr is None:
            print("Erro ao carregar imagem para OCR.")
            return None

        # Configuração para o Tesseract
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        
        # Realiza OCR na imagem e exibe o resultado
        saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)
        
        # Limita o resultado para 7 caracteres e remove espaços em branco
        resultado_limitado = saida.strip()[:7]

        print(f"Resultado OCR: {resultado_limitado}")
        return resultado_limitado

    def normalizar_texto(self, texto):
        #lista de caracteres que podem ser semelhantes
        substituicoes = {
            'Q': 'O',
            '8': 'B',
            'J': 'I',
            'O': '0',
            'I': 'l',
            '1': 'l',
            '5': 'S',
            '2': 'Z',
            '3': 'E',
            '4': 'A',
            '6': 'G',
            '9': 'G',
            '7': 'T',
            # Adicione outras substituições conforme necessário
        }
    
        texto_normalizado = ''.join(substituicoes.get(c, c) for c in texto)
        return texto_normalizado

    def placa_proxima(self, placa):
        placa_normalizada = self.normalizar_texto(placa)
        for p in self.lista_placas:
            p_normalizada = self.normalizar_texto(p)
            if difflib.SequenceMatcher(None, placa_normalizada, p_normalizada).ratio() > 0.8:  # Ajuste o valor de 0.8 conforme necessário
                return p
        return None

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            height, width = frame.shape[:2]
            top_left_x = int(width * 0.25)
            bottom_right_x = int(width * 0.75)
            top_left_y = int(height * 0.7)

            area = frame[top_left_y:height, top_left_x:bottom_right_x]

            if area.size != 0:
                # Processamento da área para melhorar visualização
                img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
                _, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)
                img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

                contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.desenhaContornoMaiorArea(contornos, area)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(10, self.update_video)

    def concluir(self):
        resultado = self.preProcessamentoRoi()
        if resultado:
            placa_encontrada = self.placa_proxima(resultado)
            if placa_encontrada:
                dfResultado = df[['regiao', 'produto']]
                index = self.lista_placas.index(placa_encontrada)
                self.resultado_label.configure(
                    text=f"Resultado OCR: {placa_encontrada} \nRegião: {dfResultado.iloc[index]['regiao']} \nProduto: {dfResultado.iloc[index]['produto']}"
                )
            else:
                self.resultado_label.configure(text=f"Resultado OCR: {resultado} DADOS NÃO ENCONTRADOS")
        else:
            self.resultado_label.configure(text="Erro ao processar a imagem.")

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = WebcamApp()
    app.mainloop()
