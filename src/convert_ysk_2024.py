import os
import pandas as pd


# Proje dizinlerini ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# YSK'den indirdiğin dosyayı Excel'de açıp
# "SecimSonucIl_2024.xlsx" olarak kaydettiğinizi varsayıyoruz.
YSK_INPUT_PATH = os.path.join(RAW_DIR, "SecimSonucIl_2024.xlsx")
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DIR, "ysk_2024_il_sonuclari.csv")


def _clean_number(value):
    """
    YSK Excel'inde sayılar noktalı binlik ayırıcı ile geliyor (örn: 1.631.643).
    Bunları tam sayıya çevir.
    """
    if pd.isna(value):
        return 0
    # Excel'den str gelebilir, nokta binlik ayırıcı olabilir
    s = str(value).strip()
    # Boş veya anlamsızsa 0
    if s == "" or s.lower() in {"nan", "none"}:
        return 0
    # Binlik ayırıcıları kaldır
    s = s.replace(".", "").replace(" ", "")
    # Virgül varsa ondalık ayracı olabilir, bu durumda yuvarla
    s = s.replace(",", ".")
    try:
        num = float(s)
    except ValueError:
        return 0
    return int(round(num))


def load_ysk_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YSK dosyası bulunamadı: {path}")

    # Genelde ilk sayfada il bazlı özet olur
    # Artık normal bir .xlsx dosyası bekliyoruz
    return pd.read_excel(path, engine="openpyxl")


def transform_ysk_to_il_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beklenen giriş yapısı (örnek):
    Il Id | İl Adı | Kayıtlı Seçmen Sayısı | Oy Kullanan Seçmen Sayısı | Geçerli Oy Toplamı | AK PARTİ | İYİ PARTİ | DEM Parti | CHP
    Bir il için 2 satır:
      - 1. satır: sayılar
      - 2. satır: 'Oy Oranı'

    Biz sadece SAYI satırlarını alacağız (İl Adı 'Oy Oranı' olmayanlar).
    Ayrıca 'İLLER TOPLAMI' satırını da atıyoruz.
    """
    # Kolon adlarını normalize et
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Sadece il satırları (Oy Oranı satırlarını at)
    if "İl Adı" not in df.columns and "Il Adı" in df.columns:
        df = df.rename(columns={"Il Adı": "İl Adı"})

    if "İl Adı" not in df.columns:
        raise ValueError("Beklenen 'İl Adı' kolonu bulunamadı.")

    mask_oy_orani = df["İl Adı"].astype(str).str.contains("Oy Oranı", case=False, na=False)
    mask_toplam = df["İl Adı"].astype(str).str.contains("İLLER TOPLAMI", case=False, na=False)

    df_il = df[~mask_oy_orani & ~mask_toplam].copy()

    # Sadece şehir isimlerini temizle
    df_il["İl Adı"] = df_il["İl Adı"].astype(str).str.strip().str.upper()

    # Sayısal kolonları temizle
    numeric_cols_map = {}
    for col in df_il.columns:
        col_str = str(col).strip()
        if any(
            key in col_str
            for key in [
                "Seçmen",
                "Oy Toplamı",
                "AK PARTİ",
                "AKPARTİ",
                "AKP",
                "CHP",
                "MHP",
                "DEM",
                "İYİ PARTİ",
                "İYİPARTİ",
            ]
        ):
            numeric_cols_map[col] = col_str

    for col in numeric_cols_map.keys():
        df_il[col] = df_il[col].apply(_clean_number)

    # Çıktı şemasını projeye yakın kur
    out = pd.DataFrame()
    out["İl Adı"] = df_il["İl Adı"]

    # Seçmen / oy bilgileri (varsa)
    for src, dst in [
        ("Kayıtlı Seçmen Sayısı", "2024 Kayıtlı Seçmen Sayısı"),
        ("Oy Kullanan Seçmen Sayısı", "2024 Oy Kullanan Seçmen Sayısı"),
        ("Geçerli Oy Toplamı", "2024 Geçerli Oy Toplamı"),
    ]:
        if src in df_il.columns:
            out[dst] = df_il[src]

    # Parti oy sayıları - kolon isimleri dosyaya göre değişebilir
    # AK PARTİ
    for akp_col in ["AK PARTİ", "AKPARTİ", "AKP"]:
        if akp_col in df_il.columns:
            out["2024 AK PARTİ Oy Sayısı (Gerçek)"] = df_il[akp_col]
            break

    # CHP
    if "CHP" in df_il.columns:
        out["2024 CHP Oy Sayısı (Gerçek)"] = df_il["CHP"]

    # MHP
    if "MHP" in df_il.columns:
        out["2024 MHP Oy Sayısı (Gerçek)"] = df_il["MHP"]

    # DEM / HDP benzeri
    for dem_col in ["DEM Parti", "DEM PARTİ", "HDP", "Yeşil Sol", "YSP"]:
        if dem_col in df_il.columns:
            out["2024 DEM/HDP Oy Sayısı (Gerçek)"] = df_il[dem_col]
            break

    # İYİ PARTİ varsa ekle
    for iyi_col in ["İYİ PARTİ", "İYİPARTİ", "IYI PARTI"]:
        if iyi_col in df_il.columns:
            out["2024 İYİ PARTİ Oy Sayısı (Gerçek)"] = df_il[iyi_col]
            break

    # Toplam oy üzerinden yüzdeler (sadece oyları varsa)
    if "2024 Geçerli Oy Toplamı" in out.columns:
        total = out["2024 Geçerli Oy Toplamı"].replace(0, pd.NA)
        for party_col in [
            "2024 AK PARTİ Oy Sayısı (Gerçek)",
            "2024 CHP Oy Sayısı (Gerçek)",
            "2024 MHP Oy Sayısı (Gerçek)",
            "2024 DEM/HDP Oy Sayısı (Gerçek)",
            "2024 İYİ PARTİ Oy Sayısı (Gerçek)",
        ]:
            if party_col in out.columns:
                out[f"{party_col} (%)"] = (out[party_col] / total * 100).round(2)

    return out


def main():
    print(f"YSK dosyası okunuyor: {YSK_INPUT_PATH}")
    df_raw = load_ysk_file(YSK_INPUT_PATH)

    print("Veri dönüştürülüyor...")
    df_out = transform_ysk_to_il_results(df_raw)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"2024 YSK il sonuçları başarıyla kaydedildi: {OUTPUT_CSV_PATH}")
    print("İlk birkaç satır:")
    print(df_out.head())


if __name__ == "__main__":
    main()


