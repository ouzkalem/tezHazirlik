# Veri düzeltme scripti
# Sorun: dropna() işlemi yıldırım olmayan günleri de siliyor
# Çözüm: Sadece modelde kullanılacak sütunlardaki eksik değerleri temizle

import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data') + os.sep

# Ana veri setini yükle
ana_df = pd.read_pickle(DATA_PATH + 'ana_veri_seti.pkl')
istasyonlar = pd.read_pickle(DATA_PATH + 'istasyonlar.pkl')

print(f"Ana veri seti: {ana_df.shape}")
print(f"yildirim_var dağılımı ÖNCE:")
print(ana_df['yildirim_var'].value_counts())

# İstasyon bilgilerini ekle
ana_df = ana_df.merge(
    istasyonlar[['istasyon_no', 'enlem', 'boylam', 'rakim']],
    on='istasyon_no',
    how='left'
)

# Lag özellikleri ekle
def ekle_lag_ozellikleri(df, sutun, lag_gunleri=[1, 2, 3, 7]):
    for lag in lag_gunleri:
        df[f'{sutun}_lag{lag}'] = df.groupby('istasyon_no')[sutun].shift(lag)
    return df

ana_df = ekle_lag_ozellikleri(ana_df, 'yildirim_sayisi', [1, 2, 3, 7])
ana_df = ekle_lag_ozellikleri(ana_df, 'yildirim_var', [1, 2, 3, 7])

# Rolling özellikleri ekle
def ekle_rolling_ozellikler(df, sutun, pencereler=[3, 7, 14]):
    for pencere in pencereler:
        df[f'{sutun}_rolling_mean_{pencere}'] = df.groupby('istasyon_no')[sutun].transform(
            lambda x: x.rolling(window=pencere, min_periods=1).mean().shift(1)
        )
        df[f'{sutun}_rolling_sum_{pencere}'] = df.groupby('istasyon_no')[sutun].transform(
            lambda x: x.rolling(window=pencere, min_periods=1).sum().shift(1)
        )
    return df

ana_df = ekle_rolling_ozellikler(ana_df, 'yildirim_sayisi', [3, 7, 14])

# Döngüsel özellikler ekle
ana_df['ay_sin'] = np.sin(2 * np.pi * ana_df['ay'] / 12)
ana_df['ay_cos'] = np.cos(2 * np.pi * ana_df['ay'] / 12)
ana_df['gun_sin'] = np.sin(2 * np.pi * ana_df['yilin_gunu'] / 365)
ana_df['gun_cos'] = np.cos(2 * np.pi * ana_df['yilin_gunu'] / 365)
ana_df['hafta_sin'] = np.sin(2 * np.pi * ana_df['haftanin_gunu'] / 7)
ana_df['hafta_cos'] = np.cos(2 * np.pi * ana_df['haftanin_gunu'] / 7)

# DÜZELTME: Sadece model için kullanılacak sütunlardaki eksik değerleri temizle
# ort_akim, maks_akim, min_akim, ort_mesafe sütunları modelde kullanılmayacak
cikartilacak_sutunlar = ['tarih', 'yildirim_sayisi', 'yildirim_var', 'ort_akim', 'maks_akim', 'min_akim', 'ort_mesafe']
model_sutunlar = [col for col in ana_df.columns if col not in cikartilacak_sutunlar]

# Sadece model sütunlarındaki eksik değerleri kontrol et
print(f"\nTemizlemeden önce: {len(ana_df)} satır")

# Model sütunlarındaki eksik değerleri doldur veya satırları sil
# Lag ve rolling özelliklerdeki eksik değerleri 0 ile doldur
lag_rolling_cols = [col for col in model_sutunlar if 'lag' in col or 'rolling' in col]
for col in lag_rolling_cols:
    ana_df[col] = ana_df[col].fillna(0)

# Şimdi eksik değer kontrolü yap
eksik_kontrol = ana_df[model_sutunlar + ['yildirim_var']].isnull().sum()
print(f"\nEksik değerler (model sütunlarında):")
print(eksik_kontrol[eksik_kontrol > 0])

# Sadece model sütunlarında eksik değer olmayan satırları tut
ana_df_temiz = ana_df.dropna(subset=model_sutunlar + ['yildirim_var'])
print(f"Temizlemeden sonra: {len(ana_df_temiz)} satır")
print(f"Silinen satır: {len(ana_df) - len(ana_df_temiz)}")

print(f"\nyildirim_var dağılımı SONRA:")
print(ana_df_temiz['yildirim_var'].value_counts())

# Model veri setini kaydet
ana_df_temiz.to_pickle(DATA_PATH + 'model_veri_seti.pkl')
print("\nModel için hazır veri seti kaydedildi!")
print(f"Boyut: {ana_df_temiz.shape}")

# Özellik listesini kaydet
hedef = 'yildirim_var'
ozellikler = model_sutunlar

print(f"\nHedef değişken: {hedef}")
print(f"Özellik sayısı: {len(ozellikler)}")

with open(DATA_PATH + 'ozellik_listesi.json', 'w', encoding='utf-8') as f:
    json.dump({'hedef': hedef, 'ozellikler': ozellikler}, f, ensure_ascii=False, indent=2)

print("\nİşlem tamamlandı!")
