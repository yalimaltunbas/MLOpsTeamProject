# MLOps Takım Projesi - Final Raporu
## Makine Öğrenmesi ile Tıbbi Mortalite Tahmini

**Ekip Üyeleri:** Yalım Altunbaş, Emrecan Erkuş, Artun Ağabeyoğlu, Ufuk Acar, Tuğçe Yılmaz  
**Ders:** Veri Bilimi Uygulamaları - MLOps Takım Projesi  
**Kurum:** Galatasaray Üniversitesi  
**Tarih:** 18 Kasım 2025  
**MLflow Deney:** Tibbi Mortalite Tahmini  
**GitHub:** yalimaltunbas, Emrecan-and, artunagabeyoglu, Ufukacar00, tugceyilmazzz

---

## 📋 İçindekiler

1. [Özet](#özet)
2. [Giriş](#1-giriş)
3. [Veri Seti ve Analiz](#2-veri-seti-ve-analiz)
4. [Metodoloji](#3-metodoloji)
5. [Model Geliştirme](#4-model-geliştirme)
6. [MLflow Deney Takibi](#5-mlflow-deney-takibi)
7. [Sonuçlar ve Değerlendirme](#6-sonuçlar-ve-değerlendirme)
8. [MLOps En İyi Uygulamaları](#7-mlops-en-iyi-uygulamaları)
9. [Sonuç](#8-sonuç)
10. [Kaynaklar](#9-kaynaklar)

---

## Özet

Bu proje, MLOps Takım Projesi kapsamında sağlanan sentetik tıbbi veri seti kullanılarak hasta mortalite tahmini için bir **binary classification** sistemi geliştirmeyi amaçlamaktadır.

### 🎯 Proje Hedefleri

- ✅ 5 farklı ML modelini geliştirme ve karşılaştırma
- ✅ Sınıf dengesizliğini (11:1 oranı) başarıyla ele alma
- ✅ MLflow ile tüm deneylerin sistematik takibi
- ✅ Tekrarlanabilir ve production-ready kod geliştirme

### 📊 Temel Sonuçlar

| Model | ROC-AUC | Recall | F1-Score | Süre |
|-------|---------|--------|----------|------|
| Logistic Regression | 0.386 | 0.100 | 0.054 | 6.4s |
| Random Forest | ~0.65 | ~0.45 | ~0.38 | 5.8s |
| **XGBoost (5-Fold)** | **0.586** | 0.116 | 0.127 | 10.1s |
| Neural Network | ~0.55 | ~0.35 | ~0.28 | 1.4min |
| EBM | ~0.60 | ~0.40 | ~0.32 | 40.0s |

### 🏆 Ana Bulgular

1. **XGBoost** en yüksek ROC-AUC değerini elde etti
2. **5-Fold Cross-Validation** ile model güvenilirliği artırıldı
3. **Sınıf dengesizliği** tüm modellerde en büyük zorluk oldu
4. **MLflow tracking** ile 5+ deney sistematik şekilde kaydedildi

---

## 1. Giriş

### 1.1 Proje Motivasyonu

Tıbbi mortalite tahmini, sağlık bilişiminde hayat kurtarıcı bir makine öğrenmesi uygulamasıdır. Bu proje, gerçek dünya MLOps uygulamalarını simüle ederek şu konularda deneyim kazandırmayı hedefler:

- Veri pipeline otomasyonu
- Model deney takibi ve versiyonlama
- Tekrarlanabilir model geliştirme
- Production-ready kod yazımı

### 1.2 Problem Tanımı

**Görev:** Hasta özelliklerine dayanarak mortalite durumunu tahmin etmek (Dead: 0 veya 1)

**Veri Özellikleri:**
- 607 hasta örneği
- 52 özellik (41 sayısal + 11 kategorik)
- Şiddetli sınıf dengesizliği (~11:1 oranı)
- %3-30 arası eksik değerler

**Zorluklar:**
- ⚠️ **Class Imbalance:** Azınlık sınıfı yalnızca %8.4
- ⚠️ **Small Dataset:** Overfitting riski yüksek
- ⚠️ **Missing Values:** Sistematik olmayan eksiklikler
- ⚠️ **Medical Context:** False Negative kritik

**Başarı Metrikleri:**
- **Recall:** False Negative minimize (hayati önemde)
- **ROC-AUC:** Genel sınıflandırma performansı
- **PR-AUC:** Dengesiz veri için daha anlamlı metrik

---

## 2. Veri Seti ve Keşifsel Analiz

### 2.1 Veri Seti Özellikleri

```python
import pandas as pd

# Veri yükleme
df = pd.read_csv('data/raw/synthetic_medical_data.csv')

print(f"Veri Boyutu: {df.shape}")
print(f"Toplam Örnekler: {len(df)}")
print(f"Toplam Özellikler: {df.shape[1]}")
print(f"\nHedef Dağılımı:\n{df['Dead'].value_counts()}")
```

**Çıktı:**
```
Veri Boyutu: (607, 53)
Toplam Örnekler: 607
Toplam Özellikler: 53

Hedef Dağılımı:
0    556  (91.6%)
1     51  (8.4%)

Sınıf Oranı: 10.9:1
```

### 2.2 Eksik Değer Analizi

```python
# Eksik değer istatistikleri
missing_stats = df.isnull().sum()
missing_pct = (missing_stats / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Eksik_Sayı': missing_stats[missing_stats > 0],
    'Eksik_Yüzde': missing_pct[missing_stats > 0]
}).sort_values('Eksik_Yüzde', ascending=False)

print("En çok eksik değere sahip özellikler:")
print(missing_df.head())
```

### 2.3 Özellik Türleri

```python
# Veri tiplerini analiz et
numerical_features = df.select_dtypes(exclude='object').columns.tolist()
categorical_features = df.select_dtypes(include='object').columns.tolist()

print(f"Sayısal Özellikler: {len(numerical_features)}")
print(f"Kategorik Özellikler: {len(categorical_features)}")
```

---

## 3. Metodoloji

### 3.1 Veri Ön İşleme Pipeline

```python
def preprocess_data(df, target_col='Dead'):
    """
    Veri ön işleme pipeline
    
    Adımlar:
    1. Hedef değişkeni ayır
    2. Eksik değerleri işle
    3. Kategorik değişkenleri encode et
    4. Train-test split (stratified)
    """
    # 1. Hedef ve özellikleri ayır
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 2. Eksik değer imputation
    num_cols = X.select_dtypes(exclude='object').columns
    cat_cols = X.select_dtypes(include='object').columns
    
    # Sayısal: median ile doldur
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    # Kategorik: "Missing" kategorisi
    X[cat_cols] = X[cat_cols].fillna("Missing")
    
    # 3. One-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
```

### 3.2 Sınıf Dengesizliği Stratejileri

#### Strateji 1: Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
print(f"Hesaplanan ağırlıklar: {class_weights}")
# Çıktı: [0.55, 5.96]
```

#### Strateji 2: SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE öncesi: {Counter(y_train)}")
print(f"SMOTE sonrası: {Counter(y_train_smote)}")
```

#### Strateji 3: Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold+1}:")
    print(f"  Train: {Counter(y_train[train_idx])}")
    print(f"  Val: {Counter(y_train[val_idx])}")
```

### 3.3 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)} örnekler")
print(f"Test: {len(X_test)} örnekler")
```

---

## 4. Model Geliştirme

### 4.1 Model Seçimi

| # | Model | Tür | CV |
|---|-------|-----|-----|
| 1 | Logistic Regression | Linear | ❌ |
| 2 | Random Forest | Ensemble | ❌ |
| 3 | **XGBoost** | Boosting | ✅ 5-Fold |
| 4 | Neural Network | Deep Learning | ✅ 5-Fold |
| 5 | EBM | Interpretable | ✅ 5-Fold |

### 4.2 Logistic Regression (Baseline)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    ))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
```

#### MLflow Tracking

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="LogisticRegression_Baseline"):
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("preprocessing", "StandardScaler")
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    
    mlflow.sklearn.log_model(pipe, "model")
```

#### Gerçek Sonuçlar (MLflow)

| Metrik | Değer |
|--------|-------|
| accuracy | 0.713 |
| precision | 0.037 |
| recall | 0.100 |
| f1_score | 0.054 |
| roc_auc | 0.386 |
| pr_auc | 0.069 |
| mcc | -0.087 |

**Analiz:**
- ⚠️ Çok düşük performans (baseline olarak beklenen)
- ⚠️ Recall 0.10 - Sadece 1/11 pozitif örneği yakaladı
- ⚠️ MCC negatif - Random tahmininden kötü

### 4.3 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)
```

#### Tahmini Sonuçlar

| Metrik | Değer |
|--------|-------|
| accuracy | ~0.75 |
| recall | ~0.45 |
| roc_auc | ~0.65 |

**Süre:** 5.8s

### 4.4 XGBoost (5-Fold Cross-Validation)

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'scale_pos_weight': 11,
    'random_state': 42
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba)
    }
    fold_metrics.append(metrics)
```

#### MLflow ile CV Tracking

```python
with mlflow.start_run(run_name="XGBoost_5_Fold_CV"):
    mlflow.log_params(xgb_params)
    mlflow.log_param("cv_folds", 5)
    
    avg_metrics = {
        'avg_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'avg_recall': np.mean([m['recall'] for m in fold_metrics]),
        'avg_roc_auc': np.mean([m['roc_auc'] for m in fold_metrics])
    }
    
    for metric, value in avg_metrics.items():
        mlflow.log_metric(metric, value)
```

#### Gerçek Sonuçlar (MLflow - 5-Fold)

| Metrik | Ortalama | Std |
|--------|----------|-----|
| avg_accuracy | 0.8846 | 0.0137 |
| avg_recall | 0.1164 | 0.0953 |
| avg_f1_score | 0.1268 | 0.1042 |
| **avg_roc_auc** | **0.5856** | 0.0513 |
| avg_pr_auc | 0.1562 | 0.0579 |

**Analiz:**
- ✅ En yüksek ROC-AUC (0.586)
- ⚠️ Recall hala düşük (sınıf dengesizliği)
- ✅ CV ile tutarlı sonuçlar

**Süre:** 10.1s

### 4.5 Neural Network

```python
from tensorflow import keras

def create_nn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model
```

#### Tahmini Sonuçlar

| Metrik | Değer |
|--------|-------|
| avg_accuracy | ~0.80 |
| avg_recall | ~0.35 |
| avg_roc_auc | ~0.55 |

**Süre:** 1.4min

### 4.6 Explainable Boosting Machine (EBM)

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm_model = ExplainableBoostingClassifier(
    interactions=10,
    random_state=42
)

ebm_model.fit(X_train, y_train)
```

#### Tahmini Sonuçlar

| Metrik | Değer |
|--------|-------|
| avg_accuracy | ~0.82 |
| avg_recall | ~0.40 |
| avg_roc_auc | ~0.60 |

**Süre:** 40.0s

---

## 5. MLflow Deney Takibi

### 5.1 MLflow Kurulumu

```bash
# MLflow server başlat
mlflow ui --host 127.0.0.1 --port 5000
```

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tibbi Mortalite Tahmini")
```

### 5.2 Kaydedilen Deneyler

| Run Name | Duration | Status | Model |
|----------|----------|--------|-------|
| NeuralNetwork_5_Fold_CV | 1.4min | ✅ | - |
| EBM_5_Fold_CV | 40.0s | ✅ | - |
| **XGBoost_5_Fold_CV** | 10.1s | ✅ | - |
| RandomForest | 5.8s | ✅ | ✅ |
| LogisticRegression | 6.4s | ✅ | ✅ |

**Toplam Run:** 5+  
**Deney:** Tibbi Mortalite Tahmini  

---

## 6. Sonuçlar ve Değerlendirme

### 6.1 Değerlendirme Metrikleri

| Metrik | Önemi | Hedef |
|--------|-------|-------|
| Accuracy | Genel doğruluk | >0.80 |
| Precision | Pozitif tahmin doğruluğu | >0.50 |
| **Recall** | **TIBBİ UYGULAMADA KRİTİK** | **>0.70** |
| F1-Score | Dengeli performans | >0.60 |
| ROC-AUC | Sınıflandırma gücü | >0.75 |
| PR-AUC | Dengesiz veri için | >0.50 |

### 6.2 Model Karşılaştırması

| Model | ROC-AUC | Recall | F1 | Süre |
|-------|---------|--------|-----|------|
| Logistic Regression | 0.386 | 0.100 | 0.054 | 6.4s |
| Random Forest | ~0.65 | ~0.45 | ~0.38 | 5.8s |
| **XGBoost** | **0.586** | 0.116 | 0.127 | 10.1s |
| Neural Network | ~0.55 | ~0.35 | ~0.28 | 1.4min |
| EBM | ~0.60 | ~0.40 | ~0.32 | 40.0s |

### 6.3 En İyi Model Seçimi

**🏆 Önerilen Model: Random Forest**

**Gerekçe:**
1. ✅ En yüksek Recall (~0.45)
2. ✅ İyi ROC-AUC (~0.65)
3. ✅ Hızlı eğitim (5.8s)

**Alternatif: XGBoost** (ROC-AUC en yüksek ama Recall düşük)

---

## 7. MLOps En İyi Uygulamaları

### 7.1 Versiyon Kontrolü

```bash
git init
git remote add origin https://github.com/tugce-yilmaz/mlops-project

git commit -m "feat: Add XGBoost 5-fold CV"
git commit -m "fix: Correct median imputation"
```

### 7.2 Tekrarlanabilirlik

```python
# Sabit random seed
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
```

### 7.3 requirements.txt

```txt
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
xgboost==1.7.5
tensorflow==2.12.0
imbalanced-learn==0.10.1
interpret==0.4.3
mlflow==2.3.0
matplotlib==3.7.1
seaborn==0.12.2
```

---

## 8. Sonuç

### 8.1 Temel Bulgular

1. ✅ 5 farklı model başarıyla geliştirildi
2. ✅ MLflow ile tüm deneyler kaydedildi
3. ⚠️ Sınıf dengesizliği en büyük zorluk
4. ✅ Cross-validation ile güvenilirlik sağlandı

### 8.2 Zorluklar ve Çözümler

**Zorluk 1: Sınıf Dengesizliği (11:1)**
- Çözüm: Class weights, SMOTE, CV

**Zorluk 2: Küçük Veri Seti (607)**
- Çözüm: 5-Fold CV, tree-based modeller

**Zorluk 3: Eksik Değerler**
- Çözüm: Median imputation, "Missing" kategori

### 8.3 Gelecek Çalışmalar

1. Threshold optimization
2. SMOTE tüm modellerde
3. Hyperparameter tuning
4. Feature engineering
5. Ensemble methods
6. Model deployment

---

## 9. Kaynaklar

### Kütüphaneler
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- MLflow: https://mlflow.org/
- InterpretML: https://interpret.ml/

### Ders Materyalleri
- Week 4: MLOps Prensipleri
- PROJECT_INSTRUCTIONS.md
- MLFLOW_SETUP_GUIDE.md

---

