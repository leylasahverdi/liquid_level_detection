import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def show_img(x):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', x)
    cv.waitKey(0)
    cv.destroyAllWindows()

giris_klasoru = r"giris_klasor_path"
cikis_klasoru = r"cikis_klasor_path"
os.makedirs(cikis_klasoru, exist_ok=True)

for i, dosya_adi in enumerate(os.listdir(giris_klasoru), start=1):
    dosya_yolu = os.path.join(giris_klasoru, dosya_adi)
    if dosya_adi.lower().endswith(('.jpg', '.jpeg', '.png')):
        yol = os.path.join(giris_klasoru, dosya_adi)
        img = cv.imread(yol, cv.IMREAD_GRAYSCALE)
        if img is not None:
            ret, imgg = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
            normalized = imgg / 255
            satir_toplamlari = np.sum(normalized, axis=1)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(satir_toplamlari.reshape(-1, 1)).flatten()
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(satir_toplamlari)), satir_toplamlari)
            plt.xlabel("Satır (Y Ekseni)")
            plt.ylabel("Toplam Piksel Değeri")
            plt.title("Satır Bazlı Piksel Yoğunluğu")
            plt.tight_layout()
            plt.show()
            show_img(normalized)
            kayit_yolu = os.path.join(cikis_klasoru, f"satir_toplamlari_{i}.jpeg")
            plt.savefig(kayit_yolu)
            plt.close()
            veri_yolu = os.path.join(cikis_klasoru, f"satir_toplamlari_{i}.txt")
            np.savetxt(veri_yolu, satir_toplamlari)

        else:
            print("Yeterli çizgi bulunamadı.")
    else:
            print(f"Görüntü yüklenemedi: {dosya_yolu}")

