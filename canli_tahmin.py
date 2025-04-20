import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1️⃣ Eğittiğin modeli yükle
model = load_model("cnn_nesne_modeli.h5")  # Bu dosya aynı klasörde olmalı

# 2️⃣ Colab'de kullandığın sınıf etiketleri
class_names = ['avsar', 'bicak', 'kalem', 'parfum', 'kasik', 'defter', 'catal', 'sirma']

# 3️⃣ Webcam başlat
cap = cv2.VideoCapture(0)
img_size = (64, 64)  # Eğitimde kullandığın boyutla aynı

print("Webcam açıldı. 'q' tuşuna basarak çıkabilirsin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü modele uygun boyuta getir
    img = cv2.resize(frame, img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Sonucu ekrana yaz
    label = f"{predicted_class} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Canlı Tahmin", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
