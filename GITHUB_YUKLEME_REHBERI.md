# 🚀 GitHub'a Yükleme Rehberi

Bu rehber, projenizi GitHub'a yüklemek için gereken tüm adımları içerir.

---

## 📋 Ön Hazırlık

### 1. GitHub Hesabı Oluşturma

Eğer GitHub hesabınız yoksa:
1. [GitHub.com](https://github.com) adresine gidin
2. "Sign up" butonuna tıklayın
3. Hesabınızı oluşturun

### 2. Git Kurulumu Kontrolü

Git'in yüklü olup olmadığını kontrol edin:

```bash
git --version
```

Eğer yüklü değilse: [Git İndirme Sayfası](https://git-scm.com/downloads)

---

## 🔧 Adım Adım Yükleme

### ADIM 1: Git Repository'sini Başlatma

Proje dizininde (PowerShell veya Terminal'de):

```powershell
# Proje dizinine gidin
cd C:\Users\emre-\OneDrive\Desktop\election_prediction_system

# Git repository'sini başlatın
git init
```

### ADIM 2: Dosyaları Stage'e Ekleme

```powershell
# Tüm dosyaları ekle (venv ve diğer ignore edilenler hariç)
git add .
```

**Not:** `.gitignore` dosyası sayesinde `venv/`, `__pycache__/` gibi dosyalar otomatik olarak hariç tutulacaktır.

### ADIM 3: İlk Commit

```powershell
# İlk commit'i oluşturun
git commit -m "Initial commit: Türkiye Yerel Seçim Tahmin Sistemi"
```

### ADIM 4: GitHub'da Repository Oluşturma

1. [GitHub.com](https://github.com) adresine gidin
2. Sağ üst köşedeki **"+"** butonuna tıklayın
3. **"New repository"** seçeneğini seçin
4. Repository bilgilerini doldurun:
   - **Repository name:** `election-prediction-system` (veya istediğiniz isim)
   - **Description:** "Türkiye Yerel Seçim Tahmin Sistemi - XGBoost ile seçim sonuçları tahmini"
   - **Visibility:** Public veya Private seçin
   - **⚠️ ÖNEMLİ:** "Initialize this repository with a README" seçeneğini **İŞARETLEMEYİN**
5. **"Create repository"** butonuna tıklayın

### ADIM 5: Remote Repository Ekleme

GitHub'da repository oluşturduktan sonra, size bir URL verilecek. Örnek:
- `https://github.com/kullaniciadi/election-prediction-system.git`

Bu URL'yi kullanarak:

```powershell
# Remote repository'yi ekleyin (URL'yi kendi repository URL'nizle değiştirin)
git remote add origin https://github.com/KULLANICI_ADINIZ/REPOSITORY_ADI.git

# Remote'un doğru eklendiğini kontrol edin
git remote -v
```

### ADIM 6: Branch Adını Ayarlama (Opsiyonel)

```powershell
# Ana branch'i 'main' olarak ayarlayın (GitHub'ın yeni standardı)
git branch -M main
```

### ADIM 7: GitHub'a Push Etme

```powershell
# Dosyaları GitHub'a yükleyin
git push -u origin main
```

**İlk kez push yaparken GitHub kullanıcı adı ve şifre isteyebilir:**
- **Kullanıcı adı:** GitHub kullanıcı adınız
- **Şifre:** GitHub Personal Access Token (PAT) kullanmanız gerekebilir

---

## 🔐 GitHub Personal Access Token (PAT) Oluşturma

Eğer şifre ile push yapamıyorsanız, Personal Access Token oluşturmanız gerekir:

### Token Oluşturma Adımları:

1. GitHub'da sağ üst köşedeki profil resminize tıklayın
2. **"Settings"** seçeneğine gidin
3. Sol menüden **"Developer settings"** seçin
4. **"Personal access tokens"** > **"Tokens (classic)"** seçin
5. **"Generate new token"** > **"Generate new token (classic)"** seçin
6. Token bilgilerini doldurun:
   - **Note:** "Election Prediction System" (açıklama)
   - **Expiration:** İstediğiniz süre (örn: 90 days)
   - **Scopes:** `repo` seçeneğini işaretleyin
7. **"Generate token"** butonuna tıklayın
8. **⚠️ ÖNEMLİ:** Token'ı kopyalayın (bir daha gösterilmeyecek!)

### Token ile Push:

```powershell
# Push yaparken şifre yerine token kullanın
git push -u origin main
# Username: GitHub kullanıcı adınız
# Password: Oluşturduğunuz Personal Access Token
```

---

## 📝 Sonraki Commit'ler İçin

Projede değişiklik yaptıktan sonra:

```powershell
# Değişiklikleri kontrol edin
git status

# Değişiklikleri stage'e ekleyin
git add .

# Commit oluşturun
git commit -m "Açıklayıcı commit mesajı"

# GitHub'a push edin
git push
```

---

## ⚠️ Önemli Notlar

### Büyük Dosyalar Hakkında

`.gitignore` dosyasında şu dosyalar varsayılan olarak ignore edilmiyor:
- `data/models/*.json` (model dosyaları)
- `data/raw/*.xlsx` (Excel veri dosyaları)
- `data/processed/*.csv` (işlenmiş veriler)

**Eğer bu dosyaları yüklemek istemiyorsanız:**

`.gitignore` dosyasında ilgili satırların başındaki `#` işaretini kaldırın:

```gitignore
# Model files
data/models/*.json

# Data files
data/raw/*.xlsx
data/processed/*.csv
```

**Eğer bu dosyaları yüklemek istiyorsanız:**
- `.gitignore` dosyasında bu satırları olduğu gibi bırakın (yorum satırı olarak)
- Dosyalar GitHub'a yüklenecektir

### GitHub Dosya Boyutu Limitleri

- **Tek dosya limiti:** 100 MB
- **Repository limiti:** 1 GB (ücretsiz hesap)
- **Daha büyük dosyalar için:** Git LFS kullanın

---

## 🔄 Alternatif: GitHub Desktop Kullanımı

Eğer komut satırı yerine görsel arayüz tercih ediyorsanız:

1. [GitHub Desktop](https://desktop.github.com/) indirin ve kurun
2. GitHub hesabınızla giriş yapın
3. **"File"** > **"Add Local Repository"** seçin
4. Proje dizinini seçin
5. **"Publish repository"** butonuna tıklayın

---

## ✅ Başarı Kontrolü

GitHub'a başarıyla yükledikten sonra:

1. GitHub'da repository sayfanıza gidin
2. Dosyaların göründüğünü kontrol edin
3. README.md dosyasının düzgün göründüğünü kontrol edin

---

## 🆘 Sorun Giderme

### Problem: "fatal: not a git repository"

**Çözüm:**
```powershell
git init
```

### Problem: "remote origin already exists"

**Çözüm:**
```powershell
# Mevcut remote'u kaldırın
git remote remove origin

# Yeni remote ekleyin
git remote add origin https://github.com/KULLANICI_ADINIZ/REPOSITORY_ADI.git
```

### Problem: "authentication failed"

**Çözüm:**
- Personal Access Token kullanın (yukarıdaki PAT bölümüne bakın)
- Veya GitHub Desktop kullanın

### Problem: "large file detected"

**Çözüm:**
- `.gitignore` dosyasına büyük dosyaları ekleyin
- Veya Git LFS kullanın

---

## 📚 Ek Kaynaklar

- [Git Resmi Dokümantasyonu](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [GitHub Desktop](https://desktop.github.com/)

---

**Başarılar! 🎉**

