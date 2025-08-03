import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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
            lines = cv.HoughLinesP(morphed, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=10)

            # 2. Orijinal gri görüntüyü renkliye çevir
            color_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

            # 3. Çizgileri orijinal görüntü üzerine çiz
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil çizgi
                show_img(color_img)
            else:
                print("Hiç çizgi bulunamadı.")
            # 1. Küçük gürültüleri temizlemek için morfolojik açma (opening)
            opened = cv.morphologyEx(morphed, cv.MORPH_OPEN, kernel)

            # 2. İsteğe bağlı olarak tekrar closing uygula (kopuk çizgileri birleştirmek için)
            final = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)

            # 3. Görüntüyü göster
            show_img(final)





