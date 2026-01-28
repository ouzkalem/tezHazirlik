"""
Model Eğitimi ve Sonuç Analizi
Notebook 4 ve 5'in birleştirilmiş Python versiyonu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc, precision_recall_curve)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost yüklü değil, atlanacak.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM yüklü değil, atlanacak.")

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

print('Kütüphaneler yüklendi!')

# ============================================================================
# NOTEBOOK 4: MODEL EĞİTİMİ
# ============================================================================

print("\n" + "="*60)
print("NOTEBOOK 4: MODEL EĞİTİMİ")
print("="*60)

DATA_PATH = '../data/'

# Model veri setini yükle
df = pd.read_pickle(DATA_PATH + 'model_veri_seti.pkl')

# Özellik listesini yükle
with open(DATA_PATH + 'ozellik_listesi.json', 'r', encoding='utf-8') as f:
    ozellik_bilgi = json.load(f)

hedef = ozellik_bilgi['hedef']
ozellikler = ozellik_bilgi['ozellikler']

print(f"Veri seti boyutu: {df.shape}")
print(f"Hedef değişken: {hedef}")
print(f"Özellik sayısı: {len(ozellikler)}")

# Özellik ve hedef değişkenleri ayır
X = df[ozellikler]
y = df[hedef]

print(f"\nX boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"\nHedef değişken dağılımı:")
print(y.value_counts())
print(f"\nPozitif sınıf oranı: {y.mean()*100:.2f}%")

# Train-Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti: {X_test.shape[0]} örnek")

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scaler'ı kaydet
joblib.dump(scaler, DATA_PATH + 'scaler.pkl')
print("Ölçeklendirme tamamlandı!")

# Sonuçları saklamak için
sonuclar = {}
modeller = {}

def degerlendir_model(model, X_test, y_test, model_adi):
    """Model performansını değerlendirir."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    sonuc = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }
    
    print(f"\n=== {model_adi} ===")
    for metrik, deger in sonuc.items():
        if deger is not None:
            print(f"  {metrik}: {deger:.4f}")
    
    return sonuc, y_pred, y_pred_proba

# 4.3.1 Lojistik Regresyon
print("\n--- Lojistik Regresyon eğitiliyor... ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

sonuclar['Lojistik Regresyon'], lr_pred, lr_proba = degerlendir_model(
    lr_model, X_test_scaled, y_test, 'Lojistik Regresyon'
)
modeller['Lojistik Regresyon'] = lr_model

# 4.3.2 Random Forest
print("\n--- Random Forest eğitiliyor... ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

sonuclar['Random Forest'], rf_pred, rf_proba = degerlendir_model(
    rf_model, X_test, y_test, 'Random Forest'
)
modeller['Random Forest'] = rf_model

# 4.3.3 XGBoost
if XGB_AVAILABLE:
    print("\n--- XGBoost eğitiliyor... ---")
    
    # Sınıf dengesizliği için ağırlık hesapla
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    sonuclar['XGBoost'], xgb_pred, xgb_proba = degerlendir_model(
        xgb_model, X_test, y_test, 'XGBoost'
    )
    modeller['XGBoost'] = xgb_model
else:
    print("\nXGBoost atlandı (kütüphane yüklü değil)")

# 4.3.4 LightGBM
if LGB_AVAILABLE:
    print("\n--- LightGBM eğitiliyor... ---")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    sonuclar['LightGBM'], lgb_pred, lgb_proba = degerlendir_model(
        lgb_model, X_test, y_test, 'LightGBM'
    )
    modeller['LightGBM'] = lgb_model
else:
    print("\nLightGBM atlandı (kütüphane yüklü değil)")

# Model karşılaştırması
sonuc_df = pd.DataFrame(sonuclar).T
sonuc_df = sonuc_df.round(4)

print("\n=== MODEL KARŞILAŞTIRMASI ===")
print(sonuc_df)

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrik karşılaştırma
metrikler = ['accuracy', 'precision', 'recall', 'f1']
sonuc_df[metrikler].plot(kind='bar', ax=axes[0], colormap='viridis')
axes[0].set_title('Model Performans Metrikleri')
axes[0].set_ylabel('Değer')
axes[0].legend(loc='lower right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# ROC-AUC karşılaştırma
if 'roc_auc' in sonuc_df.columns:
    sonuc_df['roc_auc'].plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('ROC-AUC Karşılaştırması')
    axes[1].set_ylabel('AUC')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../tez_docs/figures/model_karsilastirma.png', dpi=150)
plt.show()
print("Figür kaydedildi: model_karsilastirma.png")

# Cross-Validation
en_iyi_model_adi = sonuc_df['f1'].idxmax()
en_iyi_model = modeller[en_iyi_model_adi]

print(f"\nEn iyi model (F1 skoruna göre): {en_iyi_model_adi}")

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if en_iyi_model_adi == 'Lojistik Regresyon':
    X_cv = X_train_scaled
else:
    X_cv = X_train

cv_scores = cross_val_score(en_iyi_model, X_cv, y_train, cv=cv, scoring='f1')

print(f"\n5-Fold CV Sonuçları (F1):")
print(f"  Ortalama: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")
print(f"  Fold skorları: {cv_scores}")

# Modelleri kaydet
for model_adi, model in modeller.items():
    dosya_adi = model_adi.lower().replace(' ', '_') + '_model.pkl'
    joblib.dump(model, DATA_PATH + dosya_adi)
    print(f"Kaydedildi: {dosya_adi}")

# Sonuçları kaydet
sonuc_df.to_pickle(DATA_PATH + 'model_sonuclari.pkl')
sonuc_df.to_csv(DATA_PATH + 'model_sonuclari.csv')

print("\nTüm modeller ve sonuçlar kaydedildi!")

# ============================================================================
# NOTEBOOK 5: SONUÇLARIN ANALİZİ
# ============================================================================

print("\n" + "="*60)
print("NOTEBOOK 5: SONUÇLARIN ANALİZİ")
print("="*60)

# Confusion Matrix Görselleştirmesi
def ciz_confusion_matrix(y_true, y_pred, model_adi, ax):
    """Confusion matrix çizer."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Yıldırım Yok', 'Yıldırım Var'],
                yticklabels=['Yıldırım Yok', 'Yıldırım Var'])
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    ax.set_title(f'{model_adi}')

# Tüm modeller için confusion matrix
n_models = len(modeller)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (model_adi, model) in enumerate(modeller.items()):
    if idx >= 4:
        break
    
    # Tahmin yap
    if 'Lojistik' in model_adi:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    ciz_confusion_matrix(y_test, y_pred, model_adi, axes[idx])

plt.tight_layout()
plt.savefig('../tez_docs/figures/confusion_matrices.png', dpi=150)
plt.show()
print("Figür kaydedildi: confusion_matrices.png")

# ROC Eğrileri
fig, ax = plt.subplots(figsize=(10, 8))

renkler = ['blue', 'green', 'red', 'orange']

for idx, (model_adi, model) in enumerate(modeller.items()):
    # Tahmin olasılıkları
    if 'Lojistik' in model_adi:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=renkler[idx % len(renkler)], lw=2,
            label=f'{model_adi} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele (AUC = 0.500)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Yanlış Pozitif Oranı (1 - Özgüllük)')
ax.set_ylabel('Doğru Pozitif Oranı (Duyarlılık)')
ax.set_title('ROC Eğrileri Karşılaştırması')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../tez_docs/figures/roc_curves.png', dpi=150)
plt.show()
print("Figür kaydedildi: roc_curves.png")

# Feature Importance (Random Forest)
if 'Random Forest' in modeller:
    rf_model = modeller['Random Forest']
    
    # Feature importance
    importance = pd.DataFrame({
        'ozellik': ozellikler,
        'onem': rf_model.feature_importances_
    }).sort_values('onem', ascending=False)
    
    # En önemli 15 özellik
    print("\nEn Önemli 15 Özellik (Random Forest):")
    print(importance.head(15))
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(10, 8))
    top_15 = importance.head(15)
    sns.barplot(x='onem', y='ozellik', data=top_15, palette='viridis', ax=ax)
    ax.set_xlabel('Önem Skoru')
    ax.set_ylabel('Özellik')
    ax.set_title('Random Forest - En Önemli 15 Özellik')
    plt.tight_layout()
    plt.savefig('../tez_docs/figures/feature_importance.png', dpi=150)
    plt.show()
    print("Figür kaydedildi: feature_importance.png")

# Precision-Recall Eğrileri
fig, ax = plt.subplots(figsize=(10, 8))

for idx, (model_adi, model) in enumerate(modeller.items()):
    # Tahmin olasılıkları
    if 'Lojistik' in model_adi:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # PR eğrisi
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    ax.plot(recall, precision, color=renkler[idx % len(renkler)], lw=2,
            label=f'{model_adi}')

ax.set_xlabel('Duyarlılık (Recall)')
ax.set_ylabel('Kesinlik (Precision)')
ax.set_title('Precision-Recall Eğrileri')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../tez_docs/figures/precision_recall_curves.png', dpi=150)
plt.show()
print("Figür kaydedildi: precision_recall_curves.png")

# En iyi model için detaylı rapor
print(f"\n=== En İyi Model: {en_iyi_model_adi} ===")
print()

if 'Lojistik' in en_iyi_model_adi:
    y_pred = en_iyi_model.predict(X_test_scaled)
else:
    y_pred = en_iyi_model.predict(X_test)

print(classification_report(y_test, y_pred, 
                            target_names=['Yıldırım Yok', 'Yıldırım Var']))

# LaTeX formatında tablo
latex_tablo = sonuc_df.to_latex(
    float_format='%.4f',
    caption='Model Performans Karşılaştırması',
    label='tab:model_sonuclar'
)

print("\nLaTeX Tablosu:")
print(latex_tablo)

# Kaydet
os.makedirs('../tez_docs/tables', exist_ok=True)
with open('../tez_docs/tables/model_sonuclar.tex', 'w', encoding='utf-8') as f:
    f.write(latex_tablo)

print("\nTablo kaydedildi: tez_docs/tables/model_sonuclar.tex")

# Özet ve Sonuçlar
print("\n" + "="*60)
print("YILDIRIM TAHMİNİ PROJESİ - SONUÇ ÖZETİ")
print("="*60)
print()
print(f"Veri Seti Boyutu: {len(df):,} kayıt")
print(f"Özellik Sayısı: {len(ozellikler)}")
print(f"Pozitif Sınıf Oranı: {y.mean()*100:.2f}%")
print()
print("Model Performansları (F1 Skoruna Göre Sıralı):")
print("-"*40)

for model_adi in sonuc_df.sort_values('f1', ascending=False).index:
    f1 = sonuc_df.loc[model_adi, 'f1']
    auc_val = sonuc_df.loc[model_adi, 'roc_auc']
    print(f"  {model_adi}: F1={f1:.4f}, AUC={auc_val:.4f}")

print()
print(f"En İyi Model: {en_iyi_model_adi}")
print("="*60)

# Tüm figürlerin listesi
print("\nOluşturulan Figürler:")
for f in os.listdir('../tez_docs/figures/'):
    print(f"  - {f}")

print("\n" + "="*60)
print("TÜM NOTEBOOK'LAR BAŞARIYLA TAMAMLANDI!")
print("="*60)
