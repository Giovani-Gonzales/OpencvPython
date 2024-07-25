import cv2
import pytesseract
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import Label
import pandas as pd

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
        
        # Limita o resultado para 7 caracteres
        resultado_limitado = saida[:7]
        
        print(resultado_limitado)
        return resultado_limitado

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

                contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                self.desenhaContornoMaiorArea(contornos, area)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(10, self.update_video)

    def concluir(self):
        resultado = self.preProcessamentoRoi()
        index = 0
        dfResultado = df[['regiao','produto']]
        if resultado in self.lista_placas:
            for i in self.lista_placas:
                if resultado == i:
                    self.resultado_label.configure(text=f"Resultado OCR: {resultado} \nRegião: {dfResultado.Iloc[index, 'regiao']} \nProduto: {dfResultado.Iloc[index, 'produto']}")
                    self.btn.configure(text='Confirmar', command=self.liberado)
                    btn2 = ctk.CTkButton(self, text="Recusar", command=self.recusado)
                    btn2.pack(pady=10, padx=10)
                else:
                    index = index + 1
        if resultado not in self.lista_placas:
            self.resultado_label.configure(text=f"Resultado OCR: {resultado} \nDADOS NÃO ENCONTRADOS")
        
        if resultado == '':
            self.resultado_label.configure(text=f"Resultado: DADOS NÃO ENCONTRADOS")
    def liberado(self):
        self.resultado_label.configure(text=f"LIBERADO")

    def recusado(self):
        self.resultado_label.configure(text=f"RECUSADO")
    def on_closing(self):
        self.cap.release()
        self.destroy()

    


if __name__ == "__main__":
    app = WebcamApp()
    app.mainloop()
