import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
import os

# Modeli yükle
model = load_model("cnn_nesne_modeli.h5")

# Sınıf etiketleri
class_names = ['avsar', 'bicak', 'kalem', 'parfum', 'kasik', 'defter', 'catal', 'sirma']

# Tkinter dosya seçme ekranı
root = Tk()
root.withdraw()  # pencereyi gizle
file_path = filedialog.askopenfilename(
    title="Bir görsel seç",
    filetypes=[("Görsel Dosyaları", "*.jpg *.jpeg *.png")]
)

if file_path:
    print("Seçilen dosya:", file_path)
    
    img_size = (64, 64)
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"\n📸 Tahmin edilen sınıf: {predicted_class} ({confidence*100:.2f}%)")

else:
    print("⚠️ Dosya seçilmedi.")
