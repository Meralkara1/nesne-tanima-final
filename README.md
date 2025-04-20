# cnn-nesne-tanima
 CNN tabanlı görüntü sınıflandırma projesi (avsar, kalem, kasik, parfum vb. 8 sınıf)


 # 🧠 CNN Tabanlı Görüntü Sınıflandırma Projesi

Bu proje, 8 farklı nesne sınıfını tanımak üzere geliştirilmiş bir CNN modelini içermektedir. Sınıflar: avsar, kalem, kasik, parfum, defter, bicak, catal, sirma.

## 📁 Proje İçeriği

- `tekli_tahmin.py`: Tek bir görsel üzerinden sınıflandırma yapar.
- `web_cam_denem.py`: Webcam üzerinden gerçek zamanlı sınıflandırma yapar.
- `cnn_nesne_modeli.h5`: Eğitilmiş model dosyası.
- `Untitled2.ipynb - Colab.pdf`: Modelin Colab üzerinde eğitildiği notebook'un PDF çıktısı.
from tensorflow.keras.models import load_model
model = load_model('cnn_nesne_modeli.h5')
## 🚀 Başlangıç

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install tensorflow opencv-python

## Modeli yükleyin ve tahmin yapın:

from tensorflow.keras.models import load_model
model = load_model('cnn_nesne_modeli.h5')



