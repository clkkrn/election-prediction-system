# 🗳️ Election Prediction System - Türkiye Yerel Seçim Tahmin Sistemi

Türkiye'nin 81 ili için yerel seçim sonuçlarını tahmin eden, makine öğrenmesi tabanlı profesyonel bir analiz ve görselleştirme sistemidir. XGBoost algoritması kullanılarak CHP, AK PARTİ, MHP ve HDP partilerinin oy oranları tahmin edilmektedir.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım](#-kullanım)
- [Model Eğitimi](#-model-eğitimi)
- [Dashboard](#-dashboard)
- [Senaryo Analizi](#-senaryo-analizi)
- [Teknik Detaylar](#-teknik-detaylar)
- [Sorun Giderme](#-sorun-giderme)

## ✨ Özellikler

- **🤖 Makine Öğrenmesi Tahminleri**: XGBoost algoritması ile 4 parti için (CHP, AK PARTİ, MHP, HDP) oy tahminleri
- **📊 İnteraktif Dashboard**: Streamlit tabanlı modern web arayüzü ile görselleştirme
- **📈 Senaryo Analizi**: Farklı demografik ve ekonomik senaryoların seçim sonuçlarına etkisini analiz etme
- **🗺️ Harita Görselleştirme**: Folium ile Türkiye haritası üzerinde parti dağılımlarını görüntüleme
- **🏛️ İl Bazlı Analiz**: 81 il için detaylı parti oy dağılımı analizi
- **📉 Model Performans Metrikleri**: R² skoru, RMSE ve cross-validation sonuçları

## 📁 Proje Yapısı

```
election_prediction_system/
│
├── src/                          # Kaynak kodlar
│   ├── app.py                    # Ana Streamlit dashboard
│   ├── train_model.py            # Model eğitim scripti
│   └── pages/                    # Streamlit sayfa modülleri
│       └── scenario_analysis.py  # Senaryo analizi sayfası
│
├── data/                         # Veri dosyaları
│   ├── raw/                      # Ham veriler
│   │   └── election_data.xlsx   # Ana veri seti (81 il)
│   ├── processed/                # İşlenmiş veriler
│   │   ├── predictions.csv       # Model tahmin sonuçları
│   │   └── scenarios.csv         # Senaryo analiz sonuçları
│   ├── models/                   # Eğitilmiş modeller
│   │   ├── chp_model.json       # CHP parti modeli
│   │   ├── akp_model.json       # AK PARTİ modeli
│   │   ├── mhp_model.json       # MHP modeli
│   │   └── hdp_model.json       # HDP modeli
│   └── maps/                     # Harita verileri
│       ├── turkey_cities.geojson # Türkiye il sınırları
│       └── city_coordinates.json # İl merkez koordinatları
│
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### Adım 1: Projeyi İndirin

```bash
# Proje dizinine gidin
cd election_prediction_system
```

### Adım 2: Virtual Environment Oluşturun (Önerilir)

```bash
# Virtual environment oluşturun
python -m venv venv

# Virtual environment'ı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: Veri Dosyalarını Kontrol Edin

Aşağıdaki dosyaların mevcut olduğundan emin olun:
- `data/raw/election_data.xlsx` - Ana veri seti
- `data/maps/turkey_cities.geojson` - Harita görselleştirme için
- `data/maps/city_coordinates.json` - İl koordinatları

## 🎯 Hızlı Başlangıç

### Dashboard'u Çalıştırma

```bash
streamlit run src/app.py
```

Dashboard otomatik olarak tarayıcınızda açılacaktır (genellikle `http://localhost:8501`).

### Model Eğitimi

Yeni modeller eğitmek için:

```bash
python src/train_model.py
```

Bu script:
- Excel dosyasından veriyi yükler
- XGBoost modellerini eğitir
- Model dosyalarını `data/models/` klasörüne kaydeder
- Tahmin sonuçlarını `data/processed/predictions.csv` olarak kaydeder

## 💻 Kullanım

### Dashboard Özellikleri

#### Ana Panel

1. **Parti Oy Oranları Grafiği**: Seçili şehirlerde parti oy dağılımı
2. **Türkiye Geneli Oy Dağılımı**: Pasta grafik ile genel dağılım
3. **Detaylı Şehir Analizi**: İl bazında detaylı analiz
4. **Genel İstatistikler**: Türkiye geneli ağırlıklı oy oranları

#### Senaryo Analizi

Dashboard'da senaryo analizi yapabilirsiniz:

1. **Demografik Senaryolar**:
   - 65+ Yaşlı Nüfus (Emekli) değişimi (%)
   - Kişi Başına Düşen Gelir değişimi (%)

2. **Anket Senaryoları**:
   - CHP Anket Oy Oranı değişimi (%)
   - AK PARTİ Anket Oy Oranı değişimi (%)
   - MHP Anket Oy Oranı değişimi (%)
   - HDP Anket Oy Oranı değişimi (%)

3. **Sonuçlar**:
   - Mevcut durum vs Senaryo karşılaştırması
   - İl bazlı detaylı analiz
   - Değişim oranları
   - En yüksek artış gösteren iller

### Senaryo Analizi Sayfası

Ayrı bir Streamlit sayfası olarak mevcuttur:

```bash
# Ana uygulamayı çalıştırın
streamlit run src/app.py

# Dashboard'da sol menüden "Senaryo Analizi" sayfasına gidin
```

Bu sayfa:
- Önceden hesaplanmış senaryoları gösterir
- Senaryolar arası karşılaştırma yapar
- İl bazlı detaylı analiz sunar
- Değişim istatistikleri gösterir

## 🎓 Model Eğitimi

### Kullanılan Özellikler

Model eğitimi için kullanılan temel özellikler:

- **Demografik Veriler**: Seçmen sayısı, kadın/erkek seçmen, 65+ yaşlı nüfus
- **Ekonomik Veriler**: Kişi başına düşen gelir, işsizlik oranı
- **Eğitim**: Eğitim düzeyi (Lise+)
- **Geçmiş Seçim Sonuçları**: 2009, 2014, 2019 seçim sonuçları
- **Anket Verileri**: Parti anket oy oranları
- **Kategorik Veriler**: İl adı, kazanan parti bilgileri

### Model Parametreleri

```python
XGBRegressor(
    random_state=42,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='rmse'
)
```

### Performans Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilir:
- **R² Skoru**: Model açıklama gücü (0-1 arası, 1'e yakın daha iyi)
- **RMSE**: Root Mean Squared Error (kök ortalama kare hatası)
- **Cross-Validation**: 5-fold cross-validation ile model doğrulanır

## 📊 Teknik Detaylar

### Veri İşleme

- **One-Hot Encoding**: Kategorik değişkenler için
- **Feature Selection**: Önemli özellikler manuel olarak seçilir
- **Missing Value Handling**: Eksik değerler ortalama ile doldurulur
- **Data Cleaning**: Negatif ve sonsuz değerler temizlenir

### Model Mimarisi

- **Algoritma**: XGBoost (Extreme Gradient Boosting)
- **Problem Tipi**: Regression (Regresyon)
- **Target Variables**: Her parti için ayrı model (4 model)
- **Validation**: Train-Test Split (80-20) + Cross-Validation

### Dosya Yolları

Tüm dosya yolları otomatik olarak ayarlanır:
- Veri dosyaları: `data/raw/`, `data/processed/`
- Model dosyaları: `data/models/`
- Harita dosyaları: `data/maps/`

## 🐛 Sorun Giderme

### Model Dosyaları Bulunamıyor

Eğer model dosyaları (`*.json`) bulunamıyorsa, önce model eğitimini çalıştırın:

```bash
python src/train_model.py
```

### Excel Dosyası Bulunamıyor

`data/raw/election_data.xlsx` dosyasının mevcut olduğundan emin olun.

### Dashboard Çalışmıyor

1. Tüm bağımlılıkların yüklü olduğunu kontrol edin:
   ```bash
   pip install -r requirements.txt
   ```

2. Streamlit'in doğru çalıştığını kontrol edin:
   ```bash
   streamlit --version
   ```

3. Port çakışması varsa farklı bir port kullanın:
   ```bash
   streamlit run src/app.py --server.port 8502
   ```

### Import Hataları

Eğer modül import hataları alıyorsanız:
- Virtual environment'ın aktif olduğundan emin olun
- `requirements.txt` dosyasındaki tüm paketlerin yüklü olduğunu kontrol edin

### Veri Yolu Hataları

Dosya yolları otomatik olarak ayarlanır. Eğer hata alıyorsanız:
- Proje yapısının doğru olduğundan emin olun
- `src/app.py` dosyasını proje kök dizininden çalıştırın

## 📦 Bağımlılıklar

Ana bağımlılıklar:

- **streamlit** (>=1.28.0): Web dashboard framework
- **pandas** (>=2.0.0): Veri işleme
- **numpy** (>=1.24.0): Sayısal hesaplamalar
- **xgboost** (>=2.0.0): Makine öğrenmesi modeli
- **scikit-learn** (>=1.3.0): ML araçları ve metrikler
- **plotly** (>=5.17.0): İnteraktif grafikler
- **folium** (>=0.14.0): Harita görselleştirme
- **openpyxl** (>=3.1.0): Excel dosyası okuma

Detaylı liste için `requirements.txt` dosyasına bakın.

## 📝 Notlar

- Bu tahminler makine öğrenmesi modelleri kullanılarak oluşturulmuştur
- Sonuçlar gösterge niteliğindedir ve kesinlik içermez
- Model performansı veri kalitesine ve güncelliğine bağlıdır
- Senaryo analizleri varsayımsal durumları simüle eder

## 🔄 Güncellemeler

Model performansını artırmak için:
1. Yeni veri ekleyin veya mevcut veriyi güncelleyin
2. Model parametrelerini optimize edin (`src/train_model.py`)
3. Yeni özellikler ekleyin
4. Farklı algoritmalar deneyin

## 📧 İletişim

Sorularınız veya önerileriniz için lütfen issue açın.

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlıdır.

---

**Not**: Bu proje Türkiye'nin 81 ili için yerel seçim tahminleri yapmaktadır. Sonuçlar tahminidir ve gerçek seçim sonuçlarını garanti etmez.

**Versiyon**: 1.0.0  
**Son Güncelleme**: 2024

