import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import zscore
from itertools import groupby
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
def show_img(x, winname = 'image'):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', x)
    cv.waitKey(0)
    cv.destroyAllWindows()



giris_klasoru = r"giris_klasor_path"
cikis_klasoru = r"cikis_klasor_path"

for i, dosya_adi in enumerate(os.listdir(giris_klasoru), start=1):
    dosya_yolu = os.path.join(giris_klasoru, dosya_adi)
    if dosya_adi.lower().endswith(('.jpg', '.jpeg', '.png')):
        yol = os.path.join(giris_klasoru, dosya_adi)
        img = cv.imread(yol)
        if img is not None:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (7, 7), 0)
            show_img(blur)
            sobel_y = cv.Sobel(blur, cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv.convertScaleAbs(sobel_y)  # negatifleri pozitif yap
            _, binary = cv.threshold(sobel_y, 150, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))  # geniş yatay filtre
            morphed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            show_img(morphed)
            ret, imgg = cv.threshold(morphed, 150, 255, cv.THRESH_BINARY)
            normalized = imgg / 255
            satir_toplamlari = np.sum(normalized, axis=1)
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(satir_toplamlari)), satir_toplamlari)
            plt.xlabel("Satır (Y Ekseni)")
            plt.ylabel("Toplam Piksel Değeri")
            plt.title("Satır Bazlı Piksel Yoğunluğu")
            plt.tight_layout()
            plt.show()
            show_img(normalized)
