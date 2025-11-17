# 🏥 Tıbbi Mortalite Tahmini - MLOps Projesi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3.0-green.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Makine öğrenmesi ile hasta mortalite tahmini yapan binary classification projesi. Galatasaray Üniversitesi Veri Bilimi Uygulamaları dersi kapsamında geliştirilmiştir.

**Proje Sahibi:** Yalım Altunbaş, Emrecan Erkuş, Artun Ağabeyoğlu, Ufuk Acar, Tuğçe Yılmaz 
**Tarih:** 18 Kasım 2025   
**Ders:** Veri Bilimi Uygulamaları - MLOps Takım Projesi

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Veri Seti](#-veri-seti)
- [Modeller](#-modeller)
- [Sonuçlar](#-sonuçlar)
- [MLflow Tracking](#-mlflow-tracking)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

Bu proje, sentetik tıbbi veri seti kullanarak hastaların mortalite durumunu tahmin eden bir **makine öğrenmesi sistemi** geliştirmeyi amaçlamaktadır. Proje, modern **MLOps prensiplerini** takip ederek geliştirilmiştir.

### Problem Tanımı

- **Görev:** Binary Classification (Dead: 0 = Hayatta, 1 = Vefat)
- **Veri:** 607 hasta, 52 özellik
- **Zorluk:** Şiddetli sınıf dengesizliği (11:1 oranı)
- **Hedef:** False Negative'leri minimize ederek Recall'u maksimize etmek

### Neden Bu Proje Önemli?

Tıbbi mortalite tahmini, erken uyarı sistemleri ve klinik karar destek sistemlerinde kritik rol oynar. Bu proje:

- 🏥 **Sağlık sektörü** için gerçek dünya problemini simüle eder
- 🔬 **MLOps best practices** uygular
- 📊 **Dengesiz veri** ile çalışma deneyimi sağlar
- 🚀 **Production-ready** kod geliştirme becerisi kazandırır

---

## ✨ Özellikler

### Teknik Özellikler

- ✅ **5 Farklı Model:** Logistic Regression, Random Forest, XGBoost, Neural Network, EBM
- ✅ **MLflow Integration:** Tüm deneyler otomatik loglanır
- ✅ **Cross-Validation:** 5-Fold Stratified CV ile güvenilir sonuçlar
- ✅ **Class Imbalance Handling:** SMOTE, class weights, threshold tuning
- ✅ **Reproducibility:** Sabit random seed, requirements.txt
- ✅ **Modular Code:** Clean architecture, fonksiyonel programlama

### MLOps Uygulamaları

- 📊 **Experiment Tracking:** MLflow ile tüm parametreler ve metrikler
- 🔄 **Version Control:** Git ile kod versiyonlama
- 📦 **Model Registry:** En iyi modellerin saklanması
- 🧪 **Automated Testing:** Unit testler (opsiyonel)
- 📝 **Documentation:** Detaylı README ve raporlar

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip veya conda
- Git

### Adım 1: Repository'yi Clone

```bash
git clone https://github.com/tugce-yilmaz/mlops-mortality-prediction.git
cd mlops-mortality-prediction
```

### Adım 2: Virtual Environment Oluşturma

```bash
# venv ile
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# veya conda ile
conda create -n mlops-project python=3.8
conda activate mlops-project
```

### Adım 3: Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### Adım 4: Veri Setini Hazırlama

```bash
# Sentetik veri seti oluşturma (proje kapsamında sağlanmışsa)
python generate_synthetic_data.py

# Veya mevcut veriyi kopyalama
cp path/to/synthetic_medical_data.csv data/raw/
```

---

## 💻 Kullanım

### Hızlı Başlangıç

```bash
# 1. MLflow sunucusunu başlat
mlflow ui --host 127.0.0.1 --port 5000

# 2. Yeni terminal açıp modelleri eğit
python experiments/train_all_models.py

# 3. Tarayıcıda MLflow UI'ı aç
# http://127.0.0.1:5000
```

### Tek Bir Model Eğitme

```python
# Logistic Regression
python experiments/train_logistic.py

# XGBoost (5-Fold CV)
python experiments/train_xgboost.py

# Neural Network
python experiments/train_neural_net.py
```

### Jupyter Notebook ile Keşif

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Tahmin Yapma

```python
import mlflow
import pandas as pd

# En iyi modeli yükle
model_uri = "runs:/<RUN_ID>/model"
model = mlflow.sklearn.load_model(model_uri)

# Yeni veri ile tahmin
new_data = pd.read_csv('new_patients.csv')
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

print(f"Tahminler: {predictions}")
print(f"Mortalite Olasılıkları: {probabilities}")
```

---

## 📊 Veri Seti

### Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| **Toplam Örnekler** | 607 hasta |
| **Özellik Sayısı** | 52 (41 sayısal + 11 kategorik) |
| **Hedef Değişken** | Dead (0 = Hayatta, 1 = Vefat) |
| **Sınıf Dağılımı** | 556 hayatta (91.6%), 51 vefat (8.4%) |
| **Eksik Değerler** | %3-30 arası |
| **Kaynak** | `synthetic_medical_data.csv` |

### Özellik Kategorileri

- **Demografik:** Yaş, cinsiyet, etnik köken
- **Tıbbi Geçmiş:** Tümör boyutu, hormon seviyeleri
- **Tedavi Bilgileri:** İlaç kullanımı, prosedürler
- **Laboratuvar:** Biyobelirteçler, test sonuçları

### Veri Ön İşleme

```python
# Eksik değer işleme
- Sayısal: Median imputation
- Kategorik: "Missing" kategorisi

# Encoding
- One-hot encoding (drop_first=True)

# Split
- Train: 80% (485 örnek)
- Test: 20% (122 örnek)
- Stratified sampling ile sınıf oranları korunur
```

---

## 🤖 Modeller

### 1. Logistic Regression (Baseline)

```python
Pipeline: StandardScaler + LogisticRegression
Parametreler: class_weight='balanced', max_iter=2000
```

**Performans:**
- ROC-AUC: 0.386
- Recall: 0.100
- Süre: 6.4s

### 2. Random Forest

```python
RandomForestClassifier(n_estimators=100, class_weight='balanced')
```

**Performans:**
- ROC-AUC: ~0.65
- Recall: ~0.45
- Süre: 5.8s

### 3. XGBoost (5-Fold CV) ⭐

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    scale_pos_weight=11
)
```

**Performans:**
- **ROC-AUC: 0.586** (En yüksek)
- Recall: 0.116
- Süre: 10.1s

### 4. Neural Network (5-Fold CV)

```python
Sequential([
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Performans:**
- ROC-AUC: ~0.55
- Recall: ~0.35
- Süre: 1.4min

### 5. Explainable Boosting Machine (EBM)

```python
ExplainableBoostingClassifier(interactions=10)
```

**Performans:**
- ROC-AUC: ~0.60
- Recall: ~0.40
- Süre: 40.0s

---

## 📈 Sonuçlar

### Model Karşılaştırma Tablosu

| Model | ROC-AUC | Recall | F1-Score | Accuracy | Süre |
|-------|---------|--------|----------|----------|------|
| **XGBoost (5-Fold)** | **0.586** | 0.116 | 0.127 | 0.885 | 10.1s |
| Random Forest | ~0.65 | **0.45** | 0.38 | 0.75 | 5.8s |
| EBM | ~0.60 | 0.40 | 0.32 | 0.82 | 40.0s |
| Neural Network | ~0.55 | 0.35 | 0.28 | 0.80 | 1.4min |
| Logistic Regression | 0.386 | 0.10 | 0.054 | 0.713 | 6.4s |

### Önerilen Model

🏆 **Random Forest** - En dengeli performans
- Yüksek Recall (~0.45) - FN minimize
- İyi ROC-AUC (~0.65)
- Hızlı eğitim (5.8s)

**Alternatif:** XGBoost (En yüksek ROC-AUC, threshold tuning ile iyileştirilebilir)

### Görselleştirmeler

```python
# ROC Curves
# [XGBoost için ROC curve grafiği]

# Confusion Matrix
# [Tüm modeller için confusion matrix]

# Feature Importance
# [Random Forest feature importance]
```

---

## 🔬 MLflow Tracking

### MLflow UI Erişimi

```bash
# MLflow sunucusunu başlat
mlflow ui --host 127.0.0.1 --port 5000

# Tarayıcıda aç
http://127.0.0.1:5000
```

### Kaydedilen Deneyler

**Deney Adı:** Tibbi Mortalite Tahmini  
**Tarih:** 16/11/2025, 04:21:55 PM  
**Toplam Run:** 5+

| Run Name | Duration | Status | Logged |
|----------|----------|--------|--------|
| NeuralNetwork_5_Fold_CV | 1.4min | ✅ | Metrics, Params |
| EBM_5_Fold_CV | 40.0s | ✅ | Metrics, Params |
| XGBoost_5_Fold_CV | 10.1s | ✅ | Metrics, Params, Model |
| RandomForest | 5.8s | ✅ | Metrics, Params, Model |
| LogisticRegression | 6.4s | ✅ | Metrics, Params, Model |

### MLflow Logging Örneği

```python
import mlflow

with mlflow.start_run(run_name="MyModel"):
    # Parametreler
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Metrikler
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("roc_auc", 0.75)
    
    # Model
    mlflow.sklearn.log_model(model, "model")
    
    # Artifacts
    mlflow.log_artifact("confusion_matrix.png")
```

---

## 📁 Proje Yapısı

```
mlops-mortality-prediction/
│
├── README.md                          # Bu dosya
├── PROJECT_REPORT.md                  # Detaylı proje raporu
├── requirements.txt                   # Python bağımlılıkları
├── .gitignore                         # Git ignore kuralları
│
├── data/
│   ├── raw/
│   │   └── synthetic_medical_data.csv # Ham veri
│   └── processed/
│       ├── X_train.pkl                # İşlenmiş train features
│       ├── X_test.pkl                 # İşlenmiş test features
│       ├── y_train.pkl                # Train labels
│       └── y_test.pkl                 # Test labels
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_EDA.ipynb                   # Keşifsel veri analizi
│   ├── 02_Preprocessing.ipynb         # Veri ön işleme
│   ├── 03_Baseline_Models.ipynb       # Baseline modeller
│   ├── 04_XGBoost_Tuning.ipynb        # XGBoost optimizasyonu
│   └── 05_Final_Evaluation.ipynb      # Final değerlendirme
│
├── src/                               # Kaynak kod modülleri
│   ├── __init__.py
│   ├── config.py                      # Konfigürasyon ve sabitler
│   ├── data_loader.py                 # Veri yükleme fonksiyonları
│   ├── preprocessing.py               # Ön işleme pipeline
│   ├── feature_engineering.py         # Feature engineering
│   ├── models.py                      # Model sınıfları
│   ├── evaluation.py                  # Metrik hesaplama
│   └── utils.py                       # Yardımcı fonksiyonlar
│
├── experiments/                       # Model eğitim scriptleri
│   ├── train_logistic.py              # Logistic Regression
│   ├── train_random_forest.py         # Random Forest
│   ├── train_xgboost.py               # XGBoost
│   ├── train_neural_net.py            # Neural Network
│   ├── train_ebm.py                   # EBM
│   └── train_all_models.py            # Tüm modelleri eğit
│
├── results/                           # Çıktılar
│   ├── figures/                       # Grafikler
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── feature_importance.png
│   └── reports/                       # Raporlar
│       └── final_report.pdf
│
├── models/                            # Kaydedilmiş modeller
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── tests/                             # Unit testler
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
└── mlruns/                            # MLflow artifacts
    └── 0/
        └── [experiment_runs]/
```



## 📚 Kaynaklar

### Proje Dokümantasyonu
- [Detaylı Proje Raporu](PROJECT_REPORT.md)
- [MLflow Setup Guide](docs/MLFLOW_SETUP.md)
- [Model Comparison Report](docs/MODEL_COMPARISON.md)

### Kullanılan Teknolojiler
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [InterpretML](https://interpret.ml/)

### Akademik Referanslar
- Chawla et al. (2002) - SMOTE
- Chen & Guestrin (2016) - XGBoost
- Nori et al. (2019) - InterpretML

---

