# Tıbbi Veri Seti ile Mortalite Tahmini: Makine Öğrenmesi Yaklaşımları

**Galatasaray Üniversitesi**  
**Veri Bilimi Tezsiz Yüksek Lisans Programı**  
**Veri Bilimi Uygulamaları Takım Projesi Ödevi**

---

## İçindekiler

1. [Özet](#özet)
2. [Giriş](#1-giriş)
   - 1.1 [Proje Motivasyonu](#11-proje-motivasyonu)
   - 1.2 [Problem Tanımı](#12-problem-tanımı)
   - 1.3 [Proje Hedefleri](#13-proje-hedefleri)
3. [Veri Seti](#2-veri-seti)
   - 2.1 [Veri Kaynağı](#21-veri-kaynağı)
   - 2.2 [Hedef Değişken](#22-hedef-değişken)
   - 2.3 [Değişken Türleri](#23-değişken-türleri)
   - 2.4 [Sınıf Dengesizliği](#24-sınıf-dengesizliği)
4. [Metodoloji](#3-metodoloji)
   - 3.1 [Veri Ön İşleme](#31-veri-ön-i̇şleme)
   - 3.2 [Feature Engineering](#32-feature-engineering)
   - 3.3 [Train-Test Ayrımı](#33-train-test-ayrımı)
5. [Modelleme](#4-modelleme)
   - 4.1 [Logistic Regression](#41-logistic-regression)
   - 4.2 [Random Forest](#42-random-forest)
   - 4.3 [XGBoost](#43-xgboost)
   - 4.4 [Yapay Sinir Ağı (Neural Network)](#44-yapay-sinir-ağı-neural-network)
   - 4.5 [Explainable Boosting Machine (EBM)](#45-explainable-boosting-machine-ebm)
6. [Model Değerlendirme](#5-model-değerlendirme)
   - 5.1 [Değerlendirme Metrikleri](#51-değerlendirme-metrikleri)
   - 5.2 [Model Karşılaştırması](#52-model-karşılaştırması)
7. [Sonuçlar ve Tartışma](#6-sonuçlar-ve-tartışma)
   - 6.1 [Temel Bulgular](#61-temel-bulgular)
   - 6.2 [Model Performans Analizi](#62-model-performans-analizi)
8. [Gelecek Çalışmalar](#7-gelecek-çalışmalar)
9. [Kaynaklar](#8-kaynaklar)

---

## Özet

Bu proje, sentetik tıbbi veri seti (`synthetic_medical_data`) kullanarak hastaların yaşam durumunu (Dead = 0/1) tahmin eden bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Çalışmada veri temizleme, eksik değer işlemleri, kategorik değişken kodlamaları, eğitim-test bölünmesi gibi temel veri hazırlama adımları uygulanmıştır.

**Uygulanan Modeller:**
- Logistic Regression (Baseline)
- Random Forest (Vanilla & SMOTE)
- XGBoost (En yüksek performans)
- Neural Network (Deep Learning)
- Explainable Boosting Machine (Yorumlanabilirlik odaklı)

**Ana Bulgular:**
- **XGBoost** modeli, ROC-AUC ve diğer metriklerde en yüksek performansı göstermiştir
- **SMOTE** tekniği Random Forest performansını önemli ölçüde artırmıştır
- Tıbbi sınıflandırma problemlerinde **Recall**, **PR-AUC** ve **MCC** metrikleri Accuracy'den daha kritiktir
- **Threshold optimizasyonu** ile False Negative oranı minimize edilmiştir

---

## 1. Giriş

### 1.1 Proje Motivasyonu

Tıbbi verilerde mortalite tahmini, sağlık bilişiminde kritik bir makine öğrenmesi problemidir. Özellikle klinik karar destek sistemleri, risk skorlama sistemleri ve erken müdahale sistemlerinde ölüm olasılığının tahmin edilmesi önemli rol oynar.

Geleneksel klinik skorlama sistemleri (APACHE, SOFA vb.) uzman bilgisine dayalıdır ve sınırlı sayıda değişken kullanır. Makine öğrenmesi, çok sayıda değişken arasındaki karmaşık ilişkileri yakalayarak daha güçlü tahminler yapabilir.

### 1.2 Problem Tanımı

**Ana Problem:** Verilen sentetik tıbbi veri üzerinde hastaların ölüp ölmeyeceğini (Dead=0/1) tahmin etmek.

**Zorluklar:**
- Sınıf dengesizliği (imbalanced dataset)
- Eksik veriler
- Kategorik ve sayısal değişkenlerin karışımı
- False Negative'in maliyetinin yüksek olması

### 1.3 Proje Hedefleri

Bu proje kapsamında:

1. ✅ **Veri analiz adımlarının sistematik uygulanması**
2. ✅ **Eksik verilerin uygun istatistiksel yöntemlerle işlenmesi**
3. ✅ **Kategorik değişkenlerin makine öğrenmesine uygun formata dönüştürülmesi**
4. ✅ **Dengesiz sınıf probleminin farklı yöntemlerle ele alınması**
5. ✅ **Farklı algoritmaların karşılaştırılması**
6. ✅ **En iyi modelin belirlenmesi**

amaçlanmıştır.

---

## 2. Veri Seti

### 2.1 Veri Kaynağı

Veriler `synthetic_medical_data.csv` dosyasından elde edilmiştir. Bu sentetik veri seti, gerçek tıbbi veri setlerinin özelliklerini simüle etmek üzere oluşturulmuştur.

### 2.2 Hedef Değişken

**Dead (0/1):** Hastanın hayatta olup olmadığını gösteren binary değişken.
- `0`: Hayatta
- `1`: Vefat etmiş

### 2.3 Değişken Türleri

Veri seti iki ana değişken türü içermektedir:

#### Sayısal Değişkenler
- Yaş (age)
- Test skorları
- Laboratuvar değerleri
- Vital bulgular

#### Kategorik Değişkenler
- Cinsiyet
- Etnik köken
- Eğitim seviyesi
- Tanı kodları

### 2.4 Sınıf Dengesizliği

Veriler incelendiğinde sınıflar arasında dengesizlik olduğu görülmüştür. Bu durum özellikle **Recall** ve **PR-AUC** metriklerini önemli hale getirmiştir.

---

## 3. Metodoloji

### 3.1 Veri Ön İşleme

Veri hazırlama adımları bütün projede aynı şekilde uygulanmıştır.

#### 3.1.1 Eksik Veri İşleme

**Sayısal Değişkenler:**
Eksik değerler medyan ile doldurulmuştur.

```python
num_cols = df.select_dtypes(exclude='object').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
```

**Kategorik Değişkenler:**
Eksik kategoriler "Missing" etiketi ile doldurulmuştur.

```python
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Missing")
```

**Gerekçe:**
- Medyan, outlier'lara karşı daha dayanıklıdır
- "Missing" etiketi, eksik verinin kendisinin de bir bilgi taşıyabileceğini varsayar

### 3.2 Feature Engineering

#### 3.2.1 One-Hot Encoding

Kategorik değişkenler dummy değişkenlere dönüştürülmüştür:

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

**drop_first=True** parametresi multicollinearity'yi önlemek için ilk kategoriyi referans olarak bırakır.

### 3.3 Train-Test Ayrımı

Veri %80 eğitim – %20 test şeklinde ayrılmıştır:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**stratify=y** parametresi, sınıf dengesizliğini korumak için doğru bir seçimdir.

---

## 4. Modelleme

Beş farklı model eğitilmiş ve karşılaştırılmıştır.

### 4.1 Logistic Regression

#### Model Özellikleri
- **Pipeline:** StandardScaler + LogisticRegression
- **Sınıf Dengesi:** `class_weight='balanced'`
- **Metrik:** ROC-AUC orta seviyede

#### Implementasyon

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline içinde ölçekleme + model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(
        max_iter=2000, 
        class_weight='balanced', 
        random_state=42
    ))
])

pipe.fit(X_train, y_train)
```

#### Avantajları
- Yorumlanabilir ve hızlı
- Baseline model olarak ideal

#### Dezavantajları
- Karmaşık ilişkileri yakalamada yetersiz kalır
- Doğrusal varsayımlar non-linear veriler için sınırlayıcıdır

---

### 4.2 Random Forest

İki farklı Random Forest modeli denenmiştir:

#### 4.2.1 Vanilla Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)
```

#### 4.2.2 SMOTE ile Random Forest

SMOTE (Synthetic Minority Over-sampling Technique) kullanılarak sınıf dengesizliği giderilmiştir.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_smote.fit(X_train_smote, y_train_smote)
```

#### Özellikler
- **SMOTE** kullanımı sınıf dağılımını eşitlemiştir
- Ağaç temelli model olduğu için **outlier'lara dayanıklıdır**
- Performansı Logistic Regression'dan anlamlı biçimde yüksektir

#### Avantajlar
- Non-linear ilişkileri yakalayabilir
- Feature importance sağlar
- Overfitting'e karşı ensemble yapısı ile korumalı

---

### 4.3 XGBoost

**En yüksek performansa sahip model** olarak XGBoost seçilmiştir.

#### Implementasyon

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=10,  # sınıf dengesizliğini telafi eder
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Tahmin ve threshold ayarı
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]
```

#### Threshold Optimizasyonu

```python
# Threshold 0.5 → 0.3/0.4'e çekilerek Recall artırılmıştır
threshold = 0.3
y_pred_adjusted = (y_proba >= threshold).astype(int)
```

#### XGBoost'un Avantajları
- ✅ Feature interaction yakalama
- ✅ Dengesiz veride başarılı
- ✅ Aşırı öğrenmeye karşı düzenleme (regularization)
- ✅ En yüksek AUC, güçlü öğrenme

**Bu model, proje kapsamında en güçlü adaydır.**

---

### 4.4 Yapay Sinir Ağı (Neural Network)

Keras ile 64 → 32 → 1 katmanlı bir MLP modeli kurulmuştur.

#### Model Mimarisi

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc')]
)
```

#### Eğitim

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

#### Özellikler
- **5-Fold Stratified K-Fold** ile ortalama sonuçlar elde edilmiştir
- Özellikle **PR-AUC** ve **ROC-AUC** değerleri iyidir
- Küçük veri setlerinde ANN modelleri regresyon ağaçlarına kıyasla daha az stabil olabilir

#### Avantajlar
- Esnek ve pattern öğrenir
- Büyük veri setlerinde güçlüdür

#### Dezavantajlar
- Küçük veri setlerinde overfitting riski
- Yorumlanabilirlik zayıf

---

### 4.5 Explainable Boosting Machine (EBM)

**Şeffaf** ve **açıklanabilir** yapıda bir boosting algoritmasıdır.

#### Implementasyon

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm_model = ExplainableBoostingClassifier(
    interactions=10,
    random_state=42
)

ebm_model.fit(X_train, y_train)
```

#### Özellikler
- Özellikle **sağlık sektöründe yorumlanabilirlik** açısından önemli bir alternatiftir
- Performansı iyi olmakla birlikte **XGBoost seviyesine ulaşmamıştır**

#### Avantajlar
- ✅ Her fold'un metriklerini toplar
- ✅ Precision, Recall, F1, ROC-AUC, PR-AUC hesaplanır
- ✅ Açıklanabilir tahminler

#### Kullanım Alanı
Klinik ortamlarda **"neden bu tahmin yapıldı?"** sorusuna cevap verebilmek için kritik öneme sahiptir.

---

## 5. Model Değerlendirme

### 5.1 Değerlendirme Metrikleri

Aşağıdaki değerlendirme metrikleri tüm modelleri adil bir şekilde karşılaştırmak için kullanılmıştır:

| Metrik | Açıklama | Neden Önemli? |
|--------|----------|---------------|
| **Accuracy** | Genel doğruluk oranı | Dengeli veri setlerinde anlamlı |
| **Precision** | Pozitif tahminlerin doğruluk oranı | False Positive maliyeti yüksekse kritik |
| **Recall** | Gerçek pozitifleri yakalama oranı | **Tıbbi uygulamalarda en kritik metrik** |
| **F1-Score** | Precision ve Recall'un harmonik ortalaması | Dengeli performans göstergesi |
| **ROC-AUC** | Sınıflandırıcının ayırt edebilme gücü | Threshold'dan bağımsız performans |
| **PR-AUC** | Precision-Recall eğrisi altında kalan alan | **Dengesiz veri setlerinde daha anlamlı** |
| **MCC** | Matthews Correlation Coefficient | Dengesiz sınıflarda güvenilir metrik |
| **Confusion Matrix** | TP, TN, FP, FN dağılımı | Hata türlerini gösterir |

### 5.2 Model Karşılaştırması

#### Genel Sonuçlar (Özet Tablo)

| Model | Güçlü Yanları | Zayıf Yanları | Genel Sonuç |
|-------|---------------|---------------|-------------|
| **Logistic Regression** | Basit, yorumlanabilir | Karmaşık ilişkileri kaçırır | Orta |
| **Random Forest** | Non-linear iyi | Bazen overfit | Güçlü |
| **RF + SMOTE** | Dengesizliği iyi çözülür | Veri sentetikleşir | Çok iyi |
| **XGBoost** | En yüksek AUC, güçlü öğrenme | Parametre ayarı kritik | **En iyi model** |
| **Neural Network** | Esnek, pattern öğrenir | Veri azsa overfit | İyi |
| **EBM** | Çok iyi açıklanabilirlik | Boosting gücü sınırlı | Orta-İyi |

---

## 6. Sonuçlar ve Tartışma

### 6.1 Temel Bulgular

Bu proje tıbbi bir sınıflandırma problemi olduğu için **Recall**, **PR-AUC** ve **MCC** gibi metrikler **Accuracy'den daha kritik öneme** sahiptir. 

Özellikle ölümleri kaçırmamak (**False Negative'i minimize etmek**) sağlık alanında kritik olduğundan, **XGBoost + düşük threshold yaklaşımı** uygun bir stratejidir.

### 6.2 Model Performans Analizi

#### ✅ En İyi Performans: XGBoost
- En yüksek ROC-AUC değeri
- Threshold optimizasyonu ile Recall maksimize edildi
- Parametre ayarı ile fine-tuning yapıldı

#### ✅ SMOTE'un Etkisi
- **SMOTE RF** modelinin performansını ciddi şekilde artırmıştır
- Sınıf dengesizliği problemi başarıyla çözülmüştür

#### ✅ Neural Network Potansiyeli
- **Neural Network** modeli, veri seti büyüdükçe daha da güçlü hale gelecektir
- Daha fazla veri ile deep learning yaklaşımları değerlendirilebilir

#### ✅ Açıklanabilirlik
- **EBM'nin** açıklanabilirliği, klinik ortamlarda **"neden bu tahmin yapıldı?"** sorusu için değerlidir
- XGBoost ile birlikte SHAP değerleri kullanılarak yorumlanabilirlik artırılabilir

---

## 7. Gelecek Çalışmalar

Proje kapsamında yapılabilecek iyileştirmeler:

### 7.1 Model İyileştirmeleri
- [ ] **Hyperparameter Tuning**: GridSearchCV veya Optuna ile daha detaylı parametre optimizasyonu
- [ ] **Ensemble Yöntemleri**: XGBoost + Random Forest + Neural Network stacking yaklaşımı
- [ ] **Feature Selection**: Recursive Feature Elimination (RFE) ile özellik seçimi

### 7.2 Veri İyileştirmeleri
- [ ] **Feature Engineering**: Yeni türetilmiş özellikler (yaş grupları, risk skorları)
- [ ] **Veri Artırma**: Daha fazla sentetik veri üretimi
- [ ] **Outlier Analysis**: Aykırı değerlerin detaylı analizi

### 7.3 Yorumlanabilirlik
- [ ] **SHAP Values**: Model tahminlerinin açıklanması
- [ ] **LIME**: Lokal yorumlanabilirlik
- [ ] **Partial Dependence Plots**: Feature etkilerinin görselleştirilmesi

### 7.4 Deployment
- [ ] **Model Deployment**: Flask/FastAPI ile REST API oluşturma
- [ ] **Monitoring**: Model performansının production'da izlenmesi
- [ ] **A/B Testing**: Farklı modellerin klinik ortamda test edilmesi

---

## 8. Kaynaklar

### Akademik Makaleler
1. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
3. Nori, H., et al. (2019). "InterpretML: A Unified Framework for Machine Learning Interpretability"

### Kütüphaneler ve Araçlar
- **scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **imbalanced-learn**: https://imbalanced-learn.org/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **InterpretML**: https://interpret.ml/

### Veri Bilimi Kaynakları
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)
- Pattern Recognition and Machine Learning (Christopher Bishop)

---

## Ekler

### A. Kullanılan Python Paketleri

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
imbalanced-learn>=0.9.0
interpret>=0.2.7
matplotlib>=3.4.0
seaborn>=0.11.0
```

### B. Proje Yapısı

```
project/
│
├── data/
│   └── synthetic_medical_data.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── reports/
│   ├── PROJECT_REPORT.md
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

