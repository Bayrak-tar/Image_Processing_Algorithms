import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QInputDialog, QMessageBox, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ButonUygulamasi(QWidget):
    def __init__(self):
        super().__init__()

        file_name = "2017.jpg"
        current_directory = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_directory, file_name)

        self.image = cv2.imread(full_path)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Tıklanabilir Butonlar')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        v_box = QVBoxLayout()

        buttons = [
            'Resim Dışına Kenarlık Ekleme',
            'Blurlaştırma',
            'Gama Değeri',
            'Histogram',
            'adaptive_threshold',
            'Sobel ayarı',
            'Laplacian Kenar Tespiti',
            'Canny Kenar Tespiti',
            'Deriche Filtresi',
            'Harris Corner Detection',
            'Contour Detection',
            'Yüz Tespiti',
            'Watershed Segmentasyon'
        ]

        for button_text in buttons:
            button = self.create_button(button_text)
            v_box.addWidget(button)

        h_box = QHBoxLayout()
        h_box.addLayout(v_box)
        h_box.addWidget(self.image_label)

        self.setLayout(h_box)
        self.show()

    def create_button(self, text):
        button = QPushButton(text, self)
        button.clicked.connect(lambda: self.buton_tiklandi(text))
        button.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        return button 

    def buton_tiklandi(self, buton_metni):
        cv2.destroyAllWindows()

        if buton_metni == 'Resim Dışına Kenarlık Ekleme':
            self.kenarli_resim = cv2.copyMakeBorder(self.image, 10, 10, 10, 10,
                            borderType=cv2.BORDER_CONSTANT, value=[120, 12, 240])
            
        
            original_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            kenarli_resim_rgb = cv2.cvtColor(self.kenarli_resim, cv2.COLOR_BGR2RGB)
        
            # Display the original image with added border
            plt.subplot(1, 2, 1)
            plt.imshow(original_rgb)
            plt.title('Original Resim')
            plt.subplot(1, 2, 2)
            plt.imshow(kenarli_resim_rgb)
            plt.title(' Kenarlık Eklenen Resim')        
            plt.show()


        elif buton_metni == 'Blurlaştırma':
            kernel_size = 5
            blurred_image = cv2.blur(self.image, (kernel_size, kernel_size))
            median_blurred_image = cv2.medianBlur(self.image, kernel_size)
            box_filtered_image = cv2.boxFilter(self.image, -1, (kernel_size, kernel_size))
            bilateral_filtered_image = cv2.bilateralFilter(self.image, 9, 75, 75)
            gaussian_blurred_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        
            f, eksen = plt.subplots(2, 3, figsize=(15, 10))
        
            eksen[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            eksen[0, 0].set_title('Original Image')
        
            eksen[0, 1].imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
            eksen[0, 1].set_title('Blur')
        
            eksen[0, 2].imshow(cv2.cvtColor(median_blurred_image, cv2.COLOR_BGR2RGB))
            eksen[0, 2].set_title('Median Blur')
        
            eksen[1, 0].imshow(cv2.cvtColor(box_filtered_image, cv2.COLOR_BGR2RGB))
            eksen[1, 0].set_title('Box Filter')
        
            eksen[1, 1].imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB))
            eksen[1, 1].set_title('Bilateral Filter')
        
            eksen[1, 2].imshow(cv2.cvtColor(gaussian_blurred_image, cv2.COLOR_BGR2RGB))
            eksen[1, 2].set_title('Gaussian Blur')        
            plt.show()

        elif buton_metni == 'Gama Değeri':
            gamma_value, ok = QInputDialog.getDouble(self, 'Gama Değeri', 'Gama Değeri:', min=0.1)
            if ok:
                gamma_corrected_image = self.apply_gamma_correction(self.image, gamma=gamma_value)
        
                # Display the images using Matplotlib
                plt.figure(figsize=(12, 6))
        
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
                plt.title('Orjinal Resim')
        
                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))
                plt.title('Gama Düzeltme')        
                plt.show()

                
        elif buton_metni == 'Yüz Tespiti':
            self.face_detection()        

        elif buton_metni == 'Histogram':
            hist = cv2.calcHist([self.image], [0], None, [250], [0, 250])
            print("Histogram:", hist)
            plt.plot(hist)
            plt.title('Histogram')
            plt.xlabel('Pixel Değerleri')
            plt.ylabel('Frekans')
            plt.show()
            cv2.imshow("Orjinal Resim", self.image)

        elif buton_metni == 'adaptive_threshold':
            gray_image = cv2.imread("2017.jpg", cv2.IMREAD_GRAYSCALE)
            block_size = 11
            C = 2
            adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
            cv2.imshow('Adaptive Threshold', adaptive_threshold)
            cv2.imshow("Orjinal Resim", self.image)

        elif buton_metni == 'Sobel ayarı':
            # Assuming self.image is already a NumPy array representing the image
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            low_threshold = 50
            high_threshold = 150
            sonuc1 = cv2.Canny(image, low_threshold, high_threshold, L2gradient=True)
            
            f, eksen = plt.subplots(1, 2, figsize=(17, 7))
            eksen[0].imshow(image, cmap="gray")
            eksen[1].imshow(sonuc1, cmap="gray")
            plt.show()

                

        elif buton_metni == 'Laplacian Kenar Tespiti':
            ksize, ok = QInputDialog.getInt(self, 'Laplacian Kenar Tespiti', 'Kernel Boyutu (1, 3, 5, 7):', min=1)
            if ok:
                if ksize not in [1, 3, 5, 7]:
                    QMessageBox.warning(self, 'Uyarı', 'Geçersiz kernel boyutu! Sadece 1, 3, 5, 7 değerlerini kullanabilirsiniz.', QMessageBox.Ok)
                    return

                sonuc1 = cv2.Laplacian(self.image, cv2.CV_64F, ksize=ksize)
                sonuc1 = np.uint8(np.absolute(sonuc1))

                imgBlured = cv2.GaussianBlur(self.image, (3, 3), 0)
                sonuc2 = cv2.Laplacian(imgBlured, ddepth=-1, ksize=ksize)

                figsize, ok = QInputDialog.getDouble(self, 'Laplacian Kenar Tespiti', 'Figsize Değeri:', min=0.1)
                if ok:
                    f, eksen = plt.subplots(1, 3, figsize=(50,50))
                    eksen[0].imshow(self.image, cmap="gray")
                    eksen[1].imshow(sonuc1, cmap="gray")
                    eksen[2].imshow(sonuc2, cmap="gray")
                    cv2.imshow("Orjinal Resim", self.image)



        elif buton_metni == 'Canny Kenar Tespiti':
            low_threshold, ok1 = QInputDialog.getInt(self, 'Canny Kenar Tespiti', 'Low Threshold Değeri:', min=0)
            high_threshold, ok2 = QInputDialog.getInt(self, 'Canny Kenar Tespiti', 'High Threshold Değeri:', min=low_threshold)

            if ok1 and ok2:
                sonuc1 = cv2.Canny(self.image, low_threshold, high_threshold, L2gradient=True)

                f, eksen = plt.subplots(1, 2, figsize=(17, 7))
                eksen[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), cmap="gray")
                eksen[1].imshow(sonuc1, cmap="gray")


        elif buton_metni == 'Deriche Filtresi':
            image = cv2.imread("2017.jpg", cv2.IMREAD_GRAYSCALE)
            alpha = 0.5
            kernel_size = 3

            kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
            deriche_kernel_x = alpha * kx
            deriche_kernel_y = alpha * ky

            deriche_x = cv2.filter2D(image, -1, deriche_kernel_x)
            deriche_y = cv2.filter2D(image, -1, deriche_kernel_y)

            edges = np.sqrt(deriche_x ** 2 + deriche_y ** 2)

            f, eksen = plt.subplots(1, 2, figsize=(17, 7))
            eksen[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
            eksen[1].imshow(edges, cmap="gray")


        elif buton_metni == 'Harris Corner Detection':
            self.harris_corner_detection("2017.jpg")
            
        elif buton_metni == 'Contour Detection':
            self.contour_detection(self.image)  # Yeni eklenen fonksiyon
            
        elif buton_metni == 'Watershed Segmentasyon':
            self.watershed_segmentation()        

            

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            cv2.destroyAllWindows()
            sys.exit()

    def apply_gamma_correction(self, image, gamma=1.0):
        image_normalized = image / 255.0
        gamma_corrected = np.power(image_normalized, gamma)
        gamma_corrected = np.uint8(gamma_corrected * 255)
        return gamma_corrected

    

    def harris_corner_detection(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corner_quality = 0.04
        min_distance = 10
        block_size = 3
        corners = cv2.cornerHarris(gray, block_size, 3, corner_quality)
        corners = cv2.dilate(corners, None)
        img[corners > 0.01 * corners.max()] = [0, 0, 255]
        cv2.imshow("Harris Corner Detection", img)
        cv2.imshow("Orjinal Resim", self.image)
        

    def watershed_segmentation(self):
        imgOrj = cv2.imread("2017.jpg")
        imgBlr = cv2.medianBlur(imgOrj, 31)
        imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)
        ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
        ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sureFG = np.uint8(sureFG)
        unknown = cv2.subtract(sureBG, sureFG)
        markers = cv2.connectedComponents(sureFG)[1]
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(imgOrj, markers)

        contours, hierarchy= cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        imgCopy = imgOrj.copy()
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

        f, eksen = plt.subplots(3, 3, figsize=(30, 30))
        eksen[0, 0].imshow(imgOrj)
        eksen[0, 1].imshow(imgBlr)
        eksen[0, 2].imshow(imgGray)
        eksen[1, 0].imshow(imgTH)
        eksen[1, 1].imshow(imgOPN)
        eksen[1, 2].imshow(sureBG)
        eksen[2, 0].imshow(dist_transform)
        eksen[2, 1].imshow(sureFG)
        eksen[2, 2].imshow(imgCopy)

    def face_detection(self):
            cam = cv2.VideoCapture(0)
            yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
            while True: 
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yuzler = yuz_cascade.detectMultiScale(gray, 1.3, 5)
    
                for (x, y, w, h) in yuzler:
                    cv2.rectangle(frame, (x, y), (w+x, h+y), (255, 0, 0), 5)         
    
                cv2.imshow("Kamera", frame)
                if cv2.waitKey(1) == ord("q"):             
                    break  
    
            cam.release()
            cv2.destroyAllWindows()


        
    def contour_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(img)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contour Detection", contour_image)
        cv2.imshow("Orjinal Resim", img) 
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            cv2.destroyAllWindows()
            sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    uygulama = ButonUygulamasi()
    sys.exit(app.exec_())
