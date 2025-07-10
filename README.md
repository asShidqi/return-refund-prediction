# Description

Proyek ini mengembangkan sistem otomatis untuk verifikasi klaim return-refund di marketplace menggunakan teknologi AI multimodal. Sistem ini dirancang untuk mengidentifikasi klaim palsu secara dini dengan menggabungkan analisis citra dan teks.

## Workflow Sistem

![Workflow Diagram](workflow.jpg)

Diagram di atas menunjukkan alur kerja lengkap sistem kami, mulai dari pembuatan dataset hingga evaluasi model akhir. Proses dimulai dengan scraping fitur gambar dari Tokopedia, sintesis fitur teks menggunakan BLIP dan Claude, hingga anotasi label manual. Data kemudian dibagi menjadi training, validation, dan testing untuk pemodelan dan evaluasi menggunakan kombinasi model multimodal.

## Teknologi yang Digunakan

- **DINOv2 (self-DIstillation with NO labels)**: Model transformer state-of-the-art untuk pemrosesan dan representasi citra
- **Qwen3 Embedding**: Model embedding untuk representasi teks
- **Early Fusion**: Metode penggabungan multimodal untuk mengintegrasikan data citra dan teks
- **Gradient Boosting Tree**: Classifier untuk klasifikasi akhir

## Tujuan Sistem

1. **Verifikasi Otomatis**: Melakukan verifikasi dini terhadap klaim return-refund secara otomatis
2. **Deteksi Penipuan**: Mengidentifikasi klaim palsu
3. **Efisiensi Operasional**: Mengurangi waktu verifikasi manual oleh penjual dan tim customer service
4. **Penghematan Biaya**: Menurunkan biaya operasional melalui otomatisasi proses verifikasi

## Keunggulan

- **Multimodal**: Menganalisis data citra dan teks secara bersamaan untuk hasil yang lebih akurat
- **Real-time**: Proses verifikasi yang cepat untuk meningkatkan pengalaman pengguna
- **Scalable**: Dapat menangani volume klaim yang besar secara efisien

# How to use this model

Berikut adalah panduan lengkap untuk menggunakan model kami:

## 1. Instalasi Dependencies

```bash
pip install catboost torchvision transformers numpy pandas pillow joblib
```

## 2. Kode Implementasi

```python
# 1. Install dependencies
# !pip install catboost torchvision transformers numpy pandas pillow

import torch
import numpy as np
import pandas as pd
from PIL import Image
from joblib import load
from catboost import CatBoostClassifier
from transformers import AutoProcessor, AutoModel, AutoTokenizer

# 2. Load pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINOv2 for image
dino_model_name = "facebook/dinov2-large" 
dino_processor = AutoProcessor.from_pretrained(dino_model_name)
dino_model = AutoModel.from_pretrained(dino_model_name).to(device)
dino_model.eval()

# Qwen for text
text_model_name = "Qwen/Qwen3-Embedding-0.6B"  # adjust to your use case
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).to(device)
text_model.eval()

# CatBoost model
model = load("best_model.pkl")  # path to your trained model

# 3. Define embedding functions
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((600, 600))
    inputs = dino_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = dino_model(**inputs).last_hidden_state.mean(dim=1)
    return features.cpu().numpy().squeeze()

def embed_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().squeeze()

# 4. Prepare your input
img_main_path = "example_main.jpg"
img_review_path = "example_review.jpg"
caption = "This is the product description from the user."

# 5. Embed each input
embed_main = embed_image(img_main_path)
embed_review = embed_image(img_review_path)
embed_caption = embed_text(caption)

# 6. Convert each to DataFrame columns
df_img_main_embed = pd.DataFrame(embed_main.reshape(1, -1), columns=[f"img_main_{i}" for i in range(embed_main.shape[0])])
df_img_review_embed = pd.DataFrame(embed_review.reshape(1, -1), columns=[f"img_review_{i}" for i in range(embed_review.shape[0])])
df_text_embed = pd.DataFrame(embed_caption.reshape(1, -1), columns=[f"text_feat_{i}" for i in range(embed_caption.shape[0])])

# 7. Combine all into one dataset
combine_dataset = pd.concat([df_img_main_embed, df_img_review_embed, df_text_embed], axis=1)

# 8. Predict
prediction = model.predict(combine_dataset)
print("Prediction:", prediction)
```

## 3. Input yang Diperlukan

- **img_main_path**: Path ke gambar produk utama
- **img_review_path**: Path ke gambar dari review/klaim
- **caption**: Deskripsi teks dari pengguna atau klaim

## 4. Output

Model akan memberikan prediksi berupa:
- `0`: Klaim palsu
- `1`: Klaim valid
