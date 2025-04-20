import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Modeli yükle
model = load_model("cnn_nesne_modeli.h5")

# Sınıf etiketleri
class_names = ['avsar', 'bicak', 'kalem', 'parfum', 'kasik', 'defter', 'catal', 'sirma']

# Görüntüyü merkeze kırp ve boyutlandır
def preprocess_frame(frame, size=(64, 64)):
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Webcam başlat
cap = cv2.VideoCapture(0)
print("Kamera açıldı. 'q' tuşuna basarak çıkabilirsin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü ön işle
    img_array = preprocess_frame(frame)

    # Tahmin yap
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Tahmini ekrana yaz
    label = f"{predicted_class} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Göster
    cv2.imshow("İyileştirilmiş Canlı Tahmin", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
