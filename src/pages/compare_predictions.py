import os

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import folium_static


# Dosya yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MAPS_DIR = os.path.join(DATA_DIR, "maps")

PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, "predictions.csv")
YSK_PATH = os.path.join(PROCESSED_DIR, "ysk_2024_il_sonuclari.csv")
COORDINATES_PATH = os.path.join(MAPS_DIR, "city_coordinates.json")
GEOJSON_PATH = os.path.join(MAPS_DIR, "turkey_cities.geojson")


st.set_page_config(page_title="Tahmin vs Gerçek Sonuçlar", page_icon="✅", layout="wide")


@st.cache_data
def load_data():
    df_pred = pd.read_csv(PREDICTIONS_PATH)
    df_real = pd.read_csv(YSK_PATH)

    # İl adlarını normalize et (büyük harf, trim)
    df_pred["İl Adı"] = df_pred["İl Adı"].astype(str).str.strip().str.upper()
    df_real["İl Adı"] = df_real["İl Adı"].astype(str).str.strip().str.upper()

    # Birleştir
    df = df_pred.merge(df_real, on="İl Adı", how="inner", suffixes=("_Tahmin", "_Gerçek"))

    # Gerçek oy oranlarını YSK CSV'deki yüzdeler yerine
    # parti oy toplamlarından yeniden hesapla (daha güvenilir)
    real_party_cols = [
        "2024 AK PARTİ Oy Sayısı (Gerçek)",
        "2024 CHP Oy Sayısı (Gerçek)",
        "2024 MHP Oy Sayısı (Gerçek)",
        "2024 DEM/HDP Oy Sayısı (Gerçek)",
    ]
    for col in real_party_cols:
        if col not in df.columns:
            raise KeyError(f"Beklenen kolon bulunamadı: {col}")

    total_real = df[real_party_cols].sum(axis=1)
    total_real = total_real.replace(0, np.nan)

    df["Gerçek AK PARTİ (%)"] = df["2024 AK PARTİ Oy Sayısı (Gerçek)"] / total_real * 100
    df["Gerçek CHP (%)"] = df["2024 CHP Oy Sayısı (Gerçek)"] / total_real * 100
    df["Gerçek MHP (%)"] = df["2024 MHP Oy Sayısı (Gerçek)"] / total_real * 100
    df["Gerçek HDP/DEM (%)"] = df["2024 DEM/HDP Oy Sayısı (Gerçek)"] / total_real * 100

    # NaN'leri 0'a çek (örneğin total_real=0 olan satırlar)
    for col in ["Gerçek AK PARTİ (%)", "Gerçek CHP (%)", "Gerçek MHP (%)", "Gerçek HDP/DEM (%)"]:
        df[col] = df[col].fillna(0).round(2)

    return df


@st.cache_data
def load_coordinates():
    import json

    with open(COORDINATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


df = load_data()
coordinates = load_coordinates()

PARTIES = ["CHP", "AK PARTİ", "MHP", "HDP"]
PARTY_COL_MAP = {
    "CHP": ("2024 CHP Tahmini Oy Sayısı", "2024 CHP Oy Sayısı (Gerçek)"),
    "AK PARTİ": ("2024 AK PARTİ Tahmini Oy Sayısı", "2024 AK PARTİ Oy Sayısı (Gerçek)"),
    "MHP": ("2024 MHP Tahmini Oy Sayısı", "2024 MHP Oy Sayısı (Gerçek)"),
    "HDP": ("2024 HDP Tahmini Oy Sayısı", "2024 DEM/HDP Oy Sayısı (Gerçek)"),
}
PARTY_COL_MAP_PCT = {
    "CHP": ("CHP (%)", "Gerçek CHP (%)"),
    "AK PARTİ": ("AK PARTİ (%)", "Gerçek AK PARTİ (%)"),
    "MHP": ("MHP (%)", "Gerçek MHP (%)"),
    "HDP": ("HDP (%)", "Gerçek HDP/DEM (%)"),
}
PARTY_COLORS = {
    "CHP": "#FF0000",
    "AK PARTİ": "#FFD700",
    "MHP": "#0000FF",
    "HDP": "#800080",
}


st.title("✅ Tahminler vs 2024 Gerçek Sonuçlar (YSK)")
st.markdown(
    "Bu sayfada model tahminleri ile YSK'nın açıkladığı 2024 il bazlı oy sonuçlarını "
    "**karşılaştırabilir**, hata oranlarını ve harita üzerinde farkları görebilirsiniz."
)
st.markdown("---")


def compute_metrics(df: pd.DataFrame, party: str):
    pred_col, real_col = PARTY_COL_MAP[party]
    y_pred = df[pred_col].values
    y_real = df[real_col].values

    # Toplam üzerinden R² ve RMSE
    # R²
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))

    # Ortalama mutlak yüzde hata (MAPE)
    eps = 1e-9
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + eps))) * 100

    return r2, rmse, mape


col_top1, col_top2 = st.columns(2)

with col_top1:
    selected_party = st.selectbox("Parti Seçin", PARTIES, index=0)

with col_top2:
    view_type = st.selectbox(
        "Görünüm",
        ["Yüzde (Oy Oranı)", "Oy Sayısı"],
        index=0,
    )

st.markdown("---")

# Genel metrikler
r2, rmse, mape = compute_metrics(df, selected_party)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("R² (Açıklama Gücü)", f"{r2:.3f}")
with col_m2:
    st.metric("RMSE (Oy Sayısı)", f"{rmse:,.0f}")
with col_m3:
    st.metric("MAPE (Ortalama Mutlak Yüzde Hata)", f"{mape:.2f}%")

st.caption(
    "R² değeri 1'e ne kadar yakınsa, tahminler gerçek sonuçları o kadar iyi açıklıyor demektir. "
    "MAPE değeri ise ortalama yüzde hata büyüklüğünü gösterir."
)

st.markdown("---")

# Grafik: Tahmin vs Gerçek (seçilen parti)
st.subheader(f"📊 {selected_party} - Tahmin vs Gerçek (İl Bazında)")

pred_col, real_col = PARTY_COL_MAP_PCT[selected_party] if view_type == "Yüzde (Oy Oranı)" else PARTY_COL_MAP[selected_party]

df_plot = df[["İl Adı", pred_col, real_col]].copy()
df_plot = df_plot.rename(
    columns={
        pred_col: "Tahmin",
        real_col: "Gerçek",
    }
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=df_plot["İl Adı"],
        y=df_plot["Gerçek"],
        name="Gerçek",
        marker_color="#444444",
    )
)
fig.add_trace(
    go.Bar(
        x=df_plot["İl Adı"],
        y=df_plot["Tahmin"],
        name="Tahmin",
        marker_color=PARTY_COLORS[selected_party],
        opacity=0.7,
    )
)
fig.update_layout(
    barmode="group",
    title=f"{selected_party} - Tahmin vs Gerçek ({'Oy Oranı (%)' if view_type == 'Yüzde (Oy Oranı)' else 'Oy Sayısı'})",
    xaxis_title="İl",
    yaxis_title="Oy Oranı (%)" if view_type == "Yüzde (Oy Oranı)" else "Oy Sayısı",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)


# İl bazlı ayrıntı tablosu
city_detail = st.selectbox(
    "İl seç (detay)",
    sorted(df["İl Adı"].unique()),
    index=0,
    key="city_detail_selectbox",
)
st.subheader(f"🏛️ {city_detail} için Detaylı Karşılaştırma")
city_row = df[df["İl Adı"] == city_detail].iloc[0]
detail_rows = []
for party in PARTIES:
    pct_pred, pct_real = PARTY_COL_MAP_PCT[party]
    cnt_pred, cnt_real = PARTY_COL_MAP[party]
    detail_rows.append(
        {
            "Parti": party,
            "Tahmin Oy Sayısı": int(round(city_row[cnt_pred])),
            "Gerçek Oy Sayısı": int(round(city_row[cnt_real])),
            "Tahmin Oy Oranı (%)": round(city_row[pct_pred], 2),
            "Gerçek Oy Oranı (%)": round(city_row[pct_real], 2),
            "Fark (Yüzde Puan)": round(city_row[pct_pred] - city_row[pct_real], 2),
        }
    )
df_city = pd.DataFrame(detail_rows)
st.dataframe(df_city)

st.markdown("---")


selected_party_map = st.selectbox(
    "Harita için parti seçin",
    PARTIES,
    index=PARTIES.index(selected_party) if selected_party in PARTIES else 0,
    key="map_party_selectbox",
)

# Harita başlığı seçilen partiyle
st.subheader(f"🗺️ {selected_party_map} için Tahmin Hatası Haritası")

pct_pred_col, pct_real_col = PARTY_COL_MAP_PCT[selected_party_map]
df_map = df[["İl Adı", pct_pred_col, pct_real_col]].copy()
df_map["Hata (Yüzde Puan)"] = df_map[pct_pred_col] - df_map[pct_real_col]

# Harita merkezi
m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="OpenStreetMap")

max_abs_err = df_map["Hata (Yüzde Puan)"].abs().max() or 1.0

for _, row in df_map.iterrows():
    il = row["İl Adı"]
    if il not in coordinates:
        continue
    lat = coordinates[il]["lat"]
    lon = coordinates[il]["lon"]
    err = row["Hata (Yüzde Puan)"]

    # Hata yönüne göre renk: pozitif → yeşil (tahmin fazla), negatif → kırmızı (tahmin düşük)
    if err >= 0:
        color = "#28a745"  # yeşil
    else:
        color = "#dc3545"  # kırmızı

    radius = max(4, min(25, abs(err) / max_abs_err * 25))

    popup_html = f"""
    <div style="font-family: Arial; min-width: 220px;">
        <h4 style="margin: 5px 0;">{il}</h4>
        <p style="margin: 2px 0;"><b>{selected_party_map} Tahmin Oy Oranı:</b> {row[pct_pred_col]:.2f}%</p>
        <p style="margin: 2px 0;"><b>{selected_party_map} Gerçek Oy Oranı:</b> {row[pct_real_col]:.2f}%</p>
        <p style="margin: 2px 0;"><b>Hata (Yüzde Puan):</b> {err:+.2f}</p>
    </div>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=300),
        color=color,
        fillColor=color,
        fillOpacity=0.6,
        weight=2,
        tooltip=f"{il}: {err:+.2f} yüzde puan hata",
    ).add_to(m)

folium_static(m, width=1200, height=600)

st.caption(
    f"Yeşil daireler modelin **{selected_party_map} için o ilde fazla tahmin yaptığını**, "
    f"kırmızılar ise eksik tahmin yaptığını gösterir. Dairenin boyutu hata büyüklüğünü temsil eder."
)

# Geliştiriciler
st.markdown("---")
st.caption("Geliştirenler: Emre Çelikkıran, Sinan Sukan, Yusuf Talha Akgül, Yasin Durmaz")

