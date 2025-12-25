import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from PIL import Image
import folium
from streamlit_folium import folium_static
from sklearn.metrics import r2_score
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
import os

# Dosya yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREDICTIONS_PATH = os.path.join(DATA_DIR, 'processed', 'predictions.csv')
EXCEL_PATH = os.path.join(DATA_DIR, 'raw', 'election_data.xlsx')
COORDINATES_PATH = os.path.join(DATA_DIR, 'maps', 'city_coordinates.json')
GEOJSON_PATH = os.path.join(DATA_DIR, 'maps', 'turkey_cities.geojson')

# Sayfa yapılandırması
st.set_page_config(
    page_title="2024 Yerel Seçim Tahmin Paneli",
    page_icon="🗳️",
    layout="wide"
)

# Veriyi yükle
@st.cache_data
def load_data():
    df_predictions = pd.read_csv(PREDICTIONS_PATH)
    df_main = pd.ExcelFile(EXCEL_PATH).parse('Sheet1')
    return df_predictions, df_main

df_predictions, df_main = load_data()

# Başlık
st.title("🗳️ 2024 Yerel Seçim Tahmin Paneli")
st.markdown("---")

# Yan menü
st.sidebar.title("Filtreler")
selected_party = st.sidebar.selectbox(
    "Parti Seçin",
    ["CHP", "AK PARTİ", "MHP", "HDP"]
)

selected_cities = st.sidebar.multiselect(
    "Şehir Seçin",
    df_predictions["İl Adı"].unique(),
    default=["İSTANBUL", "ANKARA", "İZMİR"]
)

# Senaryo parametreleri
st.sidebar.markdown("---")
st.sidebar.subheader("Senaryo Parametreleri")

emekli_artis = st.sidebar.slider(
    "65+ Yaşlı Nüfus (Emekli) Değişimi (%)", 
    min_value=-50, 
    max_value=50, 
    value=0,
    help="Emekli nüfusundaki yüzdelik değişim"
)

gelir_degisim = st.sidebar.slider(
    "Kişi Başına Düşen Gelir Değişimi (%)", 
    min_value=-50, 
    max_value=50, 
    value=0,
    help="Kişi başına düşen gelirdeki yüzdelik değişim"
)

st.sidebar.markdown("### Anket Parametreleri")

chp_anket_artis = st.sidebar.slider(
    "CHP Anket Oy Oranı Değişimi (%)", 
    min_value=-20, 
    max_value=20, 
    value=0,
    help="CHP anket oy oranındaki yüzdelik değişim"
)

akp_anket_artis = st.sidebar.slider(
    "AK PARTİ Anket Oy Oranı Değişimi (%)", 
    min_value=-20, 
    max_value=20, 
    value=0,
    help="AK PARTİ anket oy oranındaki yüzdelik değişim"
)

mhp_anket_artis = st.sidebar.slider(
    "MHP Anket Oy Oranı Değişimi (%)", 
    min_value=-20, 
    max_value=20, 
    value=0,
    help="MHP anket oy oranındaki yüzdelik değişim"
)

hdp_anket_artis = st.sidebar.slider(
    "HDP Anket Oy Oranı Değişimi (%)", 
    min_value=-20, 
    max_value=20, 
    value=0,
    help="HDP anket oy oranındaki yüzdelik değişim"
)

# Ana panel düzeni
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Seçili Şehirlerde Parti Oy Oranları")
    filtered_df = df_predictions[df_predictions["İl Adı"].isin(selected_cities)]
    
    fig = px.bar(
        filtered_df,
        x="İl Adı",
        y=[f"{party} (%)" for party in ["CHP", "AK PARTİ", "MHP", "HDP"]],
        title="Parti Oy Oranları",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Türkiye Geneli Oy Dağılımı")
    
    # Ağırlıklı ortalama hesaplama
    total_votes = df_predictions['2024 CHP Tahmini Oy Sayısı'] + df_predictions['2024 AK PARTİ Tahmini Oy Sayısı'] + \
                 df_predictions['2024 MHP Tahmini Oy Sayısı'] + df_predictions['2024 HDP Tahmini Oy Sayısı']
    
    weighted_votes = {
        "CHP": (df_predictions['2024 CHP Tahmini Oy Sayısı'].sum() / total_votes.sum()) * 100,
        "AK PARTİ": (df_predictions['2024 AK PARTİ Tahmini Oy Sayısı'].sum() / total_votes.sum()) * 100,
        "MHP": (df_predictions['2024 MHP Tahmini Oy Sayısı'].sum() / total_votes.sum()) * 100,
        "HDP": (df_predictions['2024 HDP Tahmini Oy Sayısı'].sum() / total_votes.sum()) * 100
    }
    
    fig_pie = px.pie(
        values=list(weighted_votes.values()),
        names=list(weighted_votes.keys()),
        title="Türkiye Geneli Ağırlıklı Oy Dağılımı"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Harita görselleştirmesi - Her ilin kazanan partisine göre renklendirme
st.markdown("---")
st.subheader("🗺️ Türkiye Haritası - İllerin Kazanan Partileri")

# Parti renkleri
PARTI_RENKLERI = {
    "CHP": "#FF0000",      # Kırmızı
    "AK PARTİ": "#FFD700", # Sarı
    "MHP": "#0000FF",      # Mavi
    "HDP": "#800080"       # Mor
}

# Her ilin kazanan partisini belirle
def get_winning_party(row):
    """Her il için en yüksek oy oranına sahip partiyi bul"""
    parties = {
        "CHP": row['CHP (%)'],
        "AK PARTİ": row['AK PARTİ (%)'],
        "MHP": row['MHP (%)'],
        "HDP": row['HDP (%)']
    }
    return max(parties, key=parties.get)

# Kazanan partileri hesapla
df_predictions['Kazanan Parti'] = df_predictions.apply(get_winning_party, axis=1)
df_predictions['Kazanan Oy Oranı'] = df_predictions.apply(
    lambda row: row[f"{row['Kazanan Parti']} (%)"], axis=1
)

# GeoJSON dosyasını yükle
@st.cache_data
def load_geojson():
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"GeoJSON dosyası yüklenemedi: {e}")
        return None

geojson_data = load_geojson()

# Harita oluştur
m = folium.Map(
    location=[39.0, 35.0],  # Türkiye merkezi
    zoom_start=6,
    tiles='OpenStreetMap'
)

# İl adlarını normalize et (GeoJSON'daki isimlerle eşleştirmek için)
def normalize_city_name(name):
    """İl adını normalize et"""
    # Türkçe karakterleri düzelt
    replacements = {
        'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
        'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
        'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
    }
    for tr, en in replacements.items():
        name = name.replace(tr, en)
    return name.upper()

# GeoJSON ile veri dosyası arasındaki özel il adı eşleştirmeleri
GEOJSON_TO_DATA_MAPPING = {
    'Afyon': 'AFYONKARAHİSAR',
    'AFYON': 'AFYONKARAHİSAR',
    'Afyonkarahisar': 'AFYONKARAHİSAR'
}

# İl adı eşleştirme sözlüğü oluştur
city_mapping = {}
for idx, row in df_predictions.iterrows():
    il_adi = row['İl Adı']
    city_mapping[il_adi] = {
        'winning_party': row['Kazanan Parti'],
        'winning_percentage': row['Kazanan Oy Oranı'],
        'chp': row['CHP (%)'],
        'akp': row['AK PARTİ (%)'],
        'mhp': row['MHP (%)'],
        'hdp': row['HDP (%)']
    }

# GeoJSON ile choropleth harita oluştur
if geojson_data:
    # Her feature için renk belirle
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        il_adi_geojson_original = props.get('name', '') or props.get('NAME', '') or props.get('NAME_1', '')
        il_adi_geojson = il_adi_geojson_original
        
        # Özel mapping kontrolü (önce özel mapping'e bak)
        if il_adi_geojson in GEOJSON_TO_DATA_MAPPING:
            il_adi_geojson = GEOJSON_TO_DATA_MAPPING[il_adi_geojson]
        elif il_adi_geojson.upper() in GEOJSON_TO_DATA_MAPPING:
            il_adi_geojson = GEOJSON_TO_DATA_MAPPING[il_adi_geojson.upper()]
        
        # İl adını eşleştir
        matched_city = None
        for city_name in city_mapping.keys():
            # Direkt eşleşme
            if il_adi_geojson.upper() == city_name.upper():
                matched_city = city_name
                break
            # Normalize edilmiş eşleşme
            if normalize_city_name(il_adi_geojson) == normalize_city_name(city_name):
                matched_city = city_name
                break
            # Normalize edilmiş GeoJSON adı, veri dosyasındaki adın başlangıcıyla eşleşiyorsa (örn: "AFYON" -> "AFYONKARAHİSAR")
            normalized_geojson = normalize_city_name(il_adi_geojson)
            normalized_data = normalize_city_name(city_name)
            if normalized_geojson and normalized_data and normalized_data.startswith(normalized_geojson):
                matched_city = city_name
                break
        
        if matched_city and matched_city in city_mapping:
            city_data = city_mapping[matched_city]
            winning_party = city_data['winning_party']
            color = PARTI_RENKLERI.get(winning_party, '#808080')
            
            # Popup içeriği
            popup_html = f"""
            <div style="font-family: Arial; min-width: 250px;">
                <h4 style="margin: 5px 0; color: {color};">{matched_city}</h4>
                <p style="margin: 5px 0;"><b>Kazanan Parti:</b> <span style="color: {color}; font-weight: bold;">{winning_party}</span></p>
                <p style="margin: 5px 0;"><b>Oy Oranı:</b> {city_data['winning_percentage']:.2f}%</p>
                <hr style="margin: 8px 0;">
                <p style="margin: 5px 0; font-size: 12px;">
                    <b>CHP:</b> {city_data['chp']:.2f}%<br>
                    <b>AK PARTİ:</b> {city_data['akp']:.2f}%<br>
                    <b>MHP:</b> {city_data['mhp']:.2f}%<br>
                    <b>HDP:</b> {city_data['hdp']:.2f}%
                </p>
            </div>
            """
            
            # Feature'a stil ekle
            feature['properties']['fillColor'] = color
            feature['properties']['fillOpacity'] = 0.7
            feature['properties']['color'] = '#333333'
            feature['properties']['weight'] = 1.5
            feature['properties']['popup'] = popup_html
            feature['properties']['tooltip'] = f"{matched_city}: {winning_party} ({city_data['winning_percentage']:.1f}%)"
    
    # GeoJSON'ı haritaya ekle
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': feature['properties'].get('fillColor', '#808080'),
            'fillOpacity': feature['properties'].get('fillOpacity', 0.7),
            'color': feature['properties'].get('color', '#333333'),
            'weight': feature['properties'].get('weight', 1.5)
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['tooltip'],
            aliases=[''],
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['popup'],
            aliases=[''],
            localize=True
        )
    ).add_to(m)
else:
    # GeoJSON yoksa marker kullan
    @st.cache_data
    def load_coordinates():
        with open(COORDINATES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    coordinates = load_coordinates()
    
    for idx, row in df_predictions.iterrows():
        il_adi = row['İl Adı']
        winning_party = row['Kazanan Parti']
        winning_percentage = row['Kazanan Oy Oranı']
        
        if il_adi in coordinates:
            lat = coordinates[il_adi]['lat']
            lon = coordinates[il_adi]['lon']
            color = PARTI_RENKLERI.get(winning_party, '#808080')
            
            popup_html = f"""
            <div style="font-family: Arial; min-width: 250px;">
                <h4 style="margin: 5px 0; color: {color};">{il_adi}</h4>
                <p style="margin: 5px 0;"><b>Kazanan Parti:</b> <span style="color: {color}; font-weight: bold;">{winning_party}</span></p>
                <p style="margin: 5px 0;"><b>Oy Oranı:</b> {winning_percentage:.2f}%</p>
                <hr style="margin: 8px 0;">
                <p style="margin: 5px 0; font-size: 12px;">
                    <b>CHP:</b> {row['CHP (%)']:.2f}%<br>
                    <b>AK PARTİ:</b> {row['AK PARTİ (%)']:.2f}%<br>
                    <b>MHP:</b> {row['MHP (%)']:.2f}%<br>
                    <b>HDP:</b> {row['HDP (%)']:.2f}%
                </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
                tooltip=f"{il_adi}: {winning_party} ({winning_percentage:.1f}%)"
            ).add_to(m)

# Legend ekle
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 200px; height: auto; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius: 5px; padding: 10px;
     font-family: Arial;">
     <h4 style="margin: 5px 0;">Parti Renkleri</h4>
     <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#FF0000"></i> CHP</p>
     <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#FFD700"></i> AK PARTİ</p>
     <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#0000FF"></i> MHP</p>
     <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#800080"></i> HDP</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Haritayı göster
folium_static(m, width=1200, height=600)

# Harita açıklaması ve istatistikler
col_legend1, col_legend2 = st.columns(2)

with col_legend1:
    st.markdown("### 📊 Parti Renkleri")
    st.markdown("""
    - 🔴 **CHP**: Kırmızı
    - 🟡 **AK PARTİ**: Sarı  
    - 🔵 **MHP**: Mavi
    - 🟣 **HDP**: Mor
    """)

with col_legend2:
    st.markdown("### 📈 İstatistikler")
    party_counts = df_predictions['Kazanan Parti'].value_counts()
    for party, count in party_counts.items():
        color_emoji = {"CHP": "🔴", "AK PARTİ": "🟡", "MHP": "🔵", "HDP": "🟣"}.get(party, "⚪")
        st.markdown(f"{color_emoji} **{party}**: {count} il")

st.caption("Harita üzerinde her il, en yüksek oy oranına sahip partiye göre renklendirilmiştir. "
           "İllere tıklayarak detaylı bilgi görebilirsiniz.")

# Senaryo analizi bölümü - Otomatik çalışacak şekilde güncellendi
# Herhangi bir parametre değiştiğinde otomatik olarak çalışır
any_change = (emekli_artis != 0 or gelir_degisim != 0 or 
              chp_anket_artis != 0 or akp_anket_artis != 0 or 
              mhp_anket_artis != 0 or hdp_anket_artis != 0)

if any_change or st.sidebar.button("Senaryo Analizi Yap"):
    if any_change:
        st.info("📊 Senaryo parametreleri değiştirildi. Analiz otomatik olarak güncelleniyor...")
    # Veriyi hazırla
    features = [
        'Seçmen Sayısı',
        '65+ Yaşlı Nüfus (Emekli)', 'Kişi Başına Düşen Gelir',
        '2019 AK PARTİ Oy Sayısı', '2019 CHP Oy Sayısı', '2019 MHP Oy Sayısı', '2019 HDP Parti Oy Sayısı',
        'CHP Anket Oy Oranı', 'AKP Anket Oy Oranı', 'MHP Anket Oy Oranı', 'HDP Parti Anket Oy Oranı'
    ]
    
    X = df_main[features].copy()
    
    # Veri temizleme ve ön işleme
    def clean_data(df):
        # Negatif değerleri 0 yap
        df = df.clip(lower=0)
        
        # NaN değerleri doldur (pandas yeni versiyon uyumluluğu)
        df = df.ffill()
        df = df.bfill()
        df = df.fillna(0)
        
        # Sonsuz değerleri temizle
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    X = clean_data(X)
    
    # Log transform seçmen sayısı ve diğer büyük değerler (0'a epsilon ekle)
    epsilon = 1e-10
    X['Seçmen Sayısı'] = np.log1p(X['Seçmen Sayısı'] + epsilon)
    if '65+ Yaşlı Nüfus (Emekli)' in X.columns:
        X['65+ Yaşlı Nüfus (Emekli)'] = np.log1p(X['65+ Yaşlı Nüfus (Emekli)'] + epsilon)
    if 'Kişi Başına Düşen Gelir' in X.columns:
        X['Kişi Başına Düşen Gelir'] = np.log1p(X['Kişi Başına Düşen Gelir'] + epsilon)
    
    # Her parti için model eğit
    models = {}
    base_predictions = {}
    scenario_predictions = {}
    
    for party in ['CHP', 'AK PARTİ', 'MHP', 'HDP']:
        y = df_predictions[f'2024 {party} Tahmini Oy Sayısı'].copy()
        
        # Target'ı temizle
        y = clean_data(pd.DataFrame(y))[y.name]
        
        # Log transform target (0'a epsilon ekle)
        y = np.log1p(y + epsilon)
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        try:
            # Model eğitimi
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                reg_lambda=1.0,
                objective='reg:squarederror',
                validate_parameters=True
            )
            
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
            
            # Final modeli eğit
            model.fit(X_scaled, y)
            
            # Base tahminler (log-space'de)
            base_pred_log = model.predict(X_scaled)
            
            # Tahminleri orijinal ölçeğe dönüştür
            base_pred = np.expm1(base_pred_log) - epsilon
            base_pred = np.maximum(base_pred, 0)  # Negatif değerleri sıfırla
            base_predictions[party] = base_pred
            
            # Model performansını göster
            r2 = cv_scores.mean()
            r2_std = cv_scores.std()
            rmse = np.sqrt(mean_squared_error(np.expm1(y) - epsilon, base_pred))
            
            st.write(f"### {party} Model Performansı")
            st.write(f"Cross-validation R² Skoru: {r2:.4f} (±{r2_std:.4f})")
            st.write(f"RMSE: {rmse:,.0f} oy")
            
            # Özellik önemliliği
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.write(f"### {party} için Önemli Özellikler")
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title=f"{party} için Özellik Önemliliği"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Senaryo tahminleri
            X_scenario = X_scaled.copy()
            
            # Demografik değişiklikler (tüm partiler için geçerli)
            if emekli_artis != 0 and '65+ Yaşlı Nüfus (Emekli)' in X_scenario.columns:
                X_scenario['65+ Yaşlı Nüfus (Emekli)'] *= (1 + emekli_artis/100)
            
            if gelir_degisim != 0 and 'Kişi Başına Düşen Gelir' in X_scenario.columns:
                X_scenario['Kişi Başına Düşen Gelir'] *= (1 + gelir_degisim/100)
            
            # Anket değişiklikleri (tüm partiler için uygulanır)
            if chp_anket_artis != 0 and 'CHP Anket Oy Oranı' in X_scenario.columns:
                X_scenario['CHP Anket Oy Oranı'] *= (1 + chp_anket_artis/100)
            
            if akp_anket_artis != 0 and 'AKP Anket Oy Oranı' in X_scenario.columns:
                X_scenario['AKP Anket Oy Oranı'] *= (1 + akp_anket_artis/100)
            
            if mhp_anket_artis != 0 and 'MHP Anket Oy Oranı' in X_scenario.columns:
                X_scenario['MHP Anket Oy Oranı'] *= (1 + mhp_anket_artis/100)
            
            if hdp_anket_artis != 0 and 'HDP Parti Anket Oy Oranı' in X_scenario.columns:
                X_scenario['HDP Parti Anket Oy Oranı'] *= (1 + hdp_anket_artis/100)
            
            scenario_pred_log = model.predict(X_scenario)
            scenario_pred = np.expm1(scenario_pred_log) - epsilon
            scenario_pred = np.maximum(scenario_pred, 0)  # Negatif değerleri sıfırla
            scenario_predictions[party] = scenario_pred
            
        except Exception as e:
            st.error(f"{party} için model eğitimi başarısız oldu: {str(e)}")
            continue
    
    # Sonuçları göster
    # Toplam oyları hesapla
    total_base_votes = sum(base_predictions[party].sum() for party in base_predictions)
    total_scenario_votes = sum(scenario_predictions[party].sum() for party in scenario_predictions)
    
    # Yüzdeleri hesapla
    base_percentages = {
        party: (predictions.sum() / total_base_votes) * 100
        for party, predictions in base_predictions.items()
    }
    
    scenario_percentages = {
        party: (predictions.sum() / total_scenario_votes) * 100
        for party, predictions in scenario_predictions.items()
    }
    
    # Karşılaştırma grafiklerini göster
    col_results1, col_results2 = st.columns(2)
    
    with col_results1:
        st.write("### Senaryo Sonuçları - Oy Oranları")
        
        comparison_data = pd.DataFrame({
            'Parti': list(base_percentages.keys()) * 2,
            'Oy Oranı (%)': list(base_percentages.values()) + list(scenario_percentages.values()),
            'Durum': ['Mevcut'] * len(base_percentages) + ['Senaryo'] * len(scenario_percentages)
        })
        
        fig_comparison = px.bar(
            comparison_data,
            x='Parti',
            y='Oy Oranı (%)',
            color='Durum',
            barmode='group',
            title='Mevcut Durum vs Senaryo Karşılaştırması'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col_results2:
        st.write("### Değişim Oranları")
        
        for party in base_percentages.keys():
            degisim = scenario_percentages[party] - base_percentages[party]
            st.metric(
                f"{party}",
                f"{scenario_percentages[party]:.2f}%",
                f"{degisim:+.2f}%",
                delta_color="normal" if degisim == 0 else ("normal" if degisim > 0 else "inverse")
            )
    
    # İl bazlı detaylı analiz
    st.markdown("---")
    st.write("### İl Bazlı Analiz")
    
    selected_city_analysis = st.selectbox(
        "İl Seçin",
        df_predictions["İl Adı"].unique()
    )
    
    col_city1, col_city2 = st.columns(2)
    
    with col_city1:
        city_idx = df_predictions[df_predictions["İl Adı"] == selected_city_analysis].index[0]
        
        city_comparison = pd.DataFrame({
            'Parti': list(base_predictions.keys()),
            'Mevcut Oy': [base_predictions[party][city_idx] for party in base_predictions.keys()],
            'Senaryo Oy': [scenario_predictions[party][city_idx] for party in scenario_predictions.keys()]
        })
        
        city_comparison['Değişim'] = city_comparison['Senaryo Oy'] - city_comparison['Mevcut Oy']
        city_comparison['Değişim (%)'] = (city_comparison['Değişim'] / city_comparison['Mevcut Oy']) * 100
        
        st.write(f"#### {selected_city_analysis} İli Detaylı Analiz")
        st.dataframe(city_comparison.round(2))
    
    with col_city2:
        fig_city = px.bar(
            city_comparison,
            x='Parti',
            y=['Mevcut Oy', 'Senaryo Oy'],
            title=f"{selected_city_analysis} İli Oy Karşılaştırması",
            barmode='group'
        )
        st.plotly_chart(fig_city, use_container_width=True)
    
    # Senaryo haritası - Senaryo sonuçlarına göre kazanan partiler
    st.markdown("---")
    st.subheader("🗺️ Senaryo Haritası - Senaryo Sonuçlarına Göre Kazanan Partiler")
    
    # Senaryo sonuçlarından her ilin kazanan partisini hesapla
    scenario_df = pd.DataFrame({
        'İl Adı': df_predictions['İl Adı']
    })
    
    # Senaryo oy yüzdelerini hesapla
    for party in ['CHP', 'AK PARTİ', 'MHP', 'HDP']:
        scenario_df[f'{party} Senaryo Oy'] = scenario_predictions[party]
        scenario_df[f'{party} Mevcut Oy'] = base_predictions[party]
    
    # Toplam oyları hesapla
    scenario_df['Toplam Senaryo Oy'] = scenario_df[['CHP Senaryo Oy', 'AK PARTİ Senaryo Oy', 'MHP Senaryo Oy', 'HDP Senaryo Oy']].sum(axis=1)
    scenario_df['Toplam Mevcut Oy'] = scenario_df[['CHP Mevcut Oy', 'AK PARTİ Mevcut Oy', 'MHP Mevcut Oy', 'HDP Mevcut Oy']].sum(axis=1)
    
    # Senaryo yüzdelerini hesapla
    for party in ['CHP', 'AK PARTİ', 'MHP', 'HDP']:
        scenario_df[f'{party} Senaryo (%)'] = (scenario_df[f'{party} Senaryo Oy'] / scenario_df['Toplam Senaryo Oy']) * 100
    
    # Senaryo kazanan partisini belirle
    def get_scenario_winning_party(row):
        parties = {
            "CHP": row['CHP Senaryo (%)'],
            "AK PARTİ": row['AK PARTİ Senaryo (%)'],
            "MHP": row['MHP Senaryo (%)'],
            "HDP": row['HDP Senaryo (%)']
        }
        return max(parties, key=parties.get)
    
    scenario_df['Senaryo Kazanan Parti'] = scenario_df.apply(get_scenario_winning_party, axis=1)
    scenario_df['Senaryo Kazanan Oy Oranı'] = scenario_df.apply(
        lambda row: row[f"{row['Senaryo Kazanan Parti']} Senaryo (%)"], axis=1
    )
    
    # Mevcut kazanan partisini de hesapla (karşılaştırma için)
    scenario_df['Mevcut Kazanan Parti'] = df_predictions['Kazanan Parti']
    scenario_df['Mevcut Kazanan Oy Oranı'] = df_predictions['Kazanan Oy Oranı']
    
    # Senaryo haritası oluştur
    m_scenario = folium.Map(
        location=[39.0, 35.0],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Senaryo verilerini haritaya ekle
    scenario_city_mapping = {}
    for idx, row in scenario_df.iterrows():
        il_adi = row['İl Adı']
        scenario_city_mapping[il_adi] = {
            'scenario_winning_party': row['Senaryo Kazanan Parti'],
            'scenario_winning_percentage': row['Senaryo Kazanan Oy Oranı'],
            'current_winning_party': row['Mevcut Kazanan Parti'],
            'current_winning_percentage': row['Mevcut Kazanan Oy Oranı'],
            'chp_scenario': row['CHP Senaryo (%)'],
            'akp_scenario': row['AK PARTİ Senaryo (%)'],
            'mhp_scenario': row['MHP Senaryo (%)'],
            'hdp_scenario': row['HDP Senaryo (%)'],
            'chp_current': row['CHP Mevcut Oy'] / row['Toplam Mevcut Oy'] * 100,
            'akp_current': row['AK PARTİ Mevcut Oy'] / row['Toplam Mevcut Oy'] * 100,
            'mhp_current': row['MHP Mevcut Oy'] / row['Toplam Mevcut Oy'] * 100,
            'hdp_current': row['HDP Mevcut Oy'] / row['Toplam Mevcut Oy'] * 100
        }
    
    # GeoJSON ile choropleth harita oluştur
    geojson_data_scenario = load_geojson()
    
    if geojson_data_scenario:
        for feature in geojson_data_scenario.get('features', []):
            props = feature.get('properties', {})
            il_adi_geojson_original = props.get('name', '') or props.get('NAME', '') or props.get('NAME_1', '')
            il_adi_geojson = il_adi_geojson_original
            
            # Özel mapping kontrolü (önce özel mapping'e bak)
            if il_adi_geojson in GEOJSON_TO_DATA_MAPPING:
                il_adi_geojson = GEOJSON_TO_DATA_MAPPING[il_adi_geojson]
            elif il_adi_geojson.upper() in GEOJSON_TO_DATA_MAPPING:
                il_adi_geojson = GEOJSON_TO_DATA_MAPPING[il_adi_geojson.upper()]
            
            # İl adını eşleştir
            matched_city = None
            for city_name in scenario_city_mapping.keys():
                # Direkt eşleşme
                if il_adi_geojson.upper() == city_name.upper():
                    matched_city = city_name
                    break
                # Normalize edilmiş eşleşme
                if normalize_city_name(il_adi_geojson) == normalize_city_name(city_name):
                    matched_city = city_name
                    break
                # Normalize edilmiş GeoJSON adı, veri dosyasındaki adın başlangıcıyla eşleşiyorsa (örn: "AFYON" -> "AFYONKARAHİSAR")
                normalized_geojson = normalize_city_name(il_adi_geojson)
                normalized_data = normalize_city_name(city_name)
                if normalized_geojson and normalized_data and normalized_data.startswith(normalized_geojson):
                    matched_city = city_name
                    break
            
            if matched_city and matched_city in scenario_city_mapping:
                city_data = scenario_city_mapping[matched_city]
                scenario_winning_party = city_data['scenario_winning_party']
                current_winning_party = city_data['current_winning_party']
                color = PARTI_RENKLERI.get(scenario_winning_party, '#808080')
                
                # Değişim durumu
                changed = scenario_winning_party != current_winning_party
                change_indicator = "🔄" if changed else ""
                
                # Popup içeriği
                popup_html = f"""
                <div style="font-family: Arial; min-width: 280px;">
                    <h4 style="margin: 5px 0; color: {color};">{matched_city} {change_indicator}</h4>
                    <p style="margin: 5px 0;"><b>Senaryo Kazanan:</b> <span style="color: {color}; font-weight: bold;">{scenario_winning_party}</span> ({city_data['scenario_winning_percentage']:.2f}%)</p>
                    <p style="margin: 5px 0;"><b>Mevcut Kazanan:</b> {current_winning_party} ({city_data['current_winning_percentage']:.2f}%)</p>
                    <hr style="margin: 8px 0;">
                    <p style="margin: 5px 0; font-size: 11px;"><b>Senaryo Oy Oranları:</b></p>
                    <p style="margin: 2px 0; font-size: 11px;">
                        CHP: {city_data['chp_scenario']:.2f}% | AK PARTİ: {city_data['akp_scenario']:.2f}%<br>
                        MHP: {city_data['mhp_scenario']:.2f}% | HDP: {city_data['hdp_scenario']:.2f}%
                    </p>
                    <hr style="margin: 8px 0;">
                    <p style="margin: 5px 0; font-size: 11px;"><b>Mevcut Oy Oranları:</b></p>
                    <p style="margin: 2px 0; font-size: 11px;">
                        CHP: {city_data['chp_current']:.2f}% | AK PARTİ: {city_data['akp_current']:.2f}%<br>
                        MHP: {city_data['mhp_current']:.2f}% | HDP: {city_data['hdp_current']:.2f}%
                    </p>
                </div>
                """
                
                # Feature'a stil ekle
                feature['properties']['fillColor'] = color
                feature['properties']['fillOpacity'] = 0.7
                feature['properties']['color'] = '#333333'
                feature['properties']['weight'] = 1.5
                feature['properties']['popup'] = popup_html
                feature['properties']['tooltip'] = f"{matched_city}: {scenario_winning_party} ({city_data['scenario_winning_percentage']:.1f}%)"
        
        # GeoJSON'ı haritaya ekle
        folium.GeoJson(
            geojson_data_scenario,
            style_function=lambda feature: {
                'fillColor': feature['properties'].get('fillColor', '#808080'),
                'fillOpacity': feature['properties'].get('fillOpacity', 0.7),
                'color': feature['properties'].get('color', '#333333'),
                'weight': feature['properties'].get('weight', 1.5)
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['tooltip'],
                aliases=[''],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['popup'],
                aliases=[''],
                localize=True
            )
        ).add_to(m_scenario)
    else:
        # GeoJSON yoksa marker kullan
        coordinates = load_coordinates()
        
        for idx, row in scenario_df.iterrows():
            il_adi = row['İl Adı']
            scenario_winning_party = row['Senaryo Kazanan Parti']
            scenario_winning_percentage = row['Senaryo Kazanan Oy Oranı']
            current_winning_party = row['Mevcut Kazanan Parti']
            
            if il_adi in coordinates:
                lat = coordinates[il_adi]['lat']
                lon = coordinates[il_adi]['lon']
                color = PARTI_RENKLERI.get(scenario_winning_party, '#808080')
                
                changed = scenario_winning_party != current_winning_party
                change_indicator = "🔄" if changed else ""
                
                city_data = scenario_city_mapping[il_adi]
                
                popup_html = f"""
                <div style="font-family: Arial; min-width: 280px;">
                    <h4 style="margin: 5px 0; color: {color};">{il_adi} {change_indicator}</h4>
                    <p style="margin: 5px 0;"><b>Senaryo Kazanan:</b> <span style="color: {color}; font-weight: bold;">{scenario_winning_party}</span> ({scenario_winning_percentage:.2f}%)</p>
                    <p style="margin: 5px 0;"><b>Mevcut Kazanan:</b> {current_winning_party} ({city_data['current_winning_percentage']:.2f}%)</p>
                    <hr style="margin: 8px 0;">
                    <p style="margin: 5px 0; font-size: 11px;"><b>Senaryo Oy Oranları:</b></p>
                    <p style="margin: 2px 0; font-size: 11px;">
                        CHP: {city_data['chp_scenario']:.2f}% | AK PARTİ: {city_data['akp_scenario']:.2f}%<br>
                        MHP: {city_data['mhp_scenario']:.2f}% | HDP: {city_data['hdp_scenario']:.2f}%
                    </p>
                </div>
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2,
                    tooltip=f"{il_adi}: {scenario_winning_party} ({scenario_winning_percentage:.1f}%)"
                ).add_to(m_scenario)
    
    # Legend ekle
    legend_html_scenario = '''
    <div style="position: fixed; 
         bottom: 50px; right: 50px; width: 220px; height: auto; 
         background-color: white; z-index:9999; font-size:14px;
         border:2px solid grey; border-radius: 5px; padding: 10px;
         font-family: Arial;">
         <h4 style="margin: 5px 0;">Senaryo - Parti Renkleri</h4>
         <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#FF0000"></i> CHP</p>
         <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#FFD700"></i> AK PARTİ</p>
         <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#0000FF"></i> MHP</p>
         <p style="margin: 3px 0;"><i class="fa fa-circle" style="color:#800080"></i> HDP</p>
         <hr style="margin: 5px 0;">
         <p style="margin: 3px 0; font-size: 12px;">🔄 = Kazanan parti değişti</p>
    </div>
    '''
    m_scenario.get_root().html.add_child(folium.Element(legend_html_scenario))
    
    # Haritayı göster
    folium_static(m_scenario, width=1200, height=600)
    
    # Senaryo harita istatistikleri
    col_scenario_map1, col_scenario_map2 = st.columns(2)
    
    with col_scenario_map1:
        st.markdown("### 📊 Senaryo - Parti Renkleri")
        st.markdown("""
        - 🔴 **CHP**: Kırmızı
        - 🟡 **AK PARTİ**: Sarı  
        - 🔵 **MHP**: Mavi
        - 🟣 **HDP**: Mor
        """)
    
    with col_scenario_map2:
        st.markdown("### 📈 Senaryo İstatistikleri")
        scenario_party_counts = scenario_df['Senaryo Kazanan Parti'].value_counts()
        for party, count in scenario_party_counts.items():
            color_emoji = {"CHP": "🔴", "AK PARTİ": "🟡", "MHP": "🔵", "HDP": "🟣"}.get(party, "⚪")
            st.markdown(f"{color_emoji} **{party}**: {count} il")
        
        # Değişen iller sayısı
        changed_cities = (scenario_df['Senaryo Kazanan Parti'] != scenario_df['Mevcut Kazanan Parti']).sum()
        st.markdown(f"🔄 **Kazanan Parti Değişen İl Sayısı**: {changed_cities}")
    
    st.caption("Senaryo haritası üzerinde her il, senaryo sonuçlarına göre en yüksek oy oranına sahip partiye göre renklendirilmiştir. "
               "🔄 işareti, kazanan partinin değiştiğini gösterir. İllere tıklayarak detaylı bilgi görebilirsiniz.")
    
    # Genel İstatistikler
    st.markdown("---")
    st.write("### Genel İstatistikler")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.write("#### Katılım Oranları")
        total_voters = df_main['Seçmen Sayısı'].sum()
        st.metric("Toplam Seçmen Sayısı", f"{total_voters:,.0f}")
        st.metric("Toplam Oy Sayısı (Mevcut)", f"{total_base_votes:,.0f}")
        st.metric("Toplam Oy Sayısı (Senaryo)", f"{total_scenario_votes:,.0f}")
    
    with col_stats2:
        st.write("#### En Yüksek Artış Gösteren İller")
        for party in base_predictions.keys():
            city_changes = pd.DataFrame({
                'İl': df_predictions['İl Adı'],
                'Değişim (%)': ((scenario_predictions[party] - base_predictions[party]) / base_predictions[party]) * 100
            }).sort_values('Değişim (%)', ascending=False)
            
            st.write(f"**{party}**")
            st.write(f"1. {city_changes.iloc[0]['İl']}: {city_changes.iloc[0]['Değişim (%)']:.2f}%")
            st.write(f"2. {city_changes.iloc[1]['İl']}: {city_changes.iloc[1]['Değişim (%)']:.2f}%")
    
    with col_stats3:
        st.write("#### Senaryo Etki Analizi")
        total_effect = abs(total_scenario_votes - total_base_votes)
        st.metric("Toplam Değişim", f"{total_effect:,.0f} oy")
        
        for param, value in {
            "Emekli Nüfus Değişimi": emekli_artis,
            "Gelir Değişimi": gelir_degisim,
            "CHP Anket Değişimi": chp_anket_artis,
            "AK PARTİ Anket Değişimi": akp_anket_artis,
            "MHP Anket Değişimi": mhp_anket_artis,
            "HDP Anket Değişimi": hdp_anket_artis
        }.items():
            if value != 0:
                st.write(f"- {param}: {value:+.1f}%")

# Model metrikleri açıklaması
st.markdown("---")
with st.expander("📊 Model performans metrikleri ne anlama geliyor?"):
    st.markdown("""
    **R-squared (R²)**  
    - Modelin oy sayılarındaki / oy oranlarındaki toplam değişimin ne kadarını açıkladığını gösterir.  
    - **0 ile 1** arasındadır, **1'e yaklaştıkça model veriyi daha iyi açıklar.**  
    - Örnek: R² = 0.92 → Değişimin yaklaşık **%92'si model tarafından açıklanıyor**, %8'i açıklanamayan kısım.

    **RMSE (Root Mean Squared Error - Kök Ortalama Kare Hatası)**  
    - Tahmin edilen oy sayıları ile gerçek oy sayıları arasındaki **ortalama hata büyüklüğünü** gösterir.  
    - Birimi oy sayısıdır; **küçük olması, tahminlerin gerçeğe daha yakın olduğunu** gösterir.

    **5-Fold Cross-Validation (Çapraz Doğrulama)**  
    - Veri 5 parçaya bölünür; her seferinde 4 parça ile model eğitilip 1 parça ile test edilir.  
    - Böylece model, **farklı veri bölünmelerinde test edilerek genelleme gücü** ölçülür.  
    - Gösterilen R² değerleri, bu 5 tekrarın ortalamasıdır.
    """)

# Footer
st.markdown("---")
st.markdown("*Bu tahminler makine öğrenmesi modelleri kullanılarak oluşturulmuştur.*")
st.markdown("*Sonuçlar gösterge niteliğindedir ve kesinlik içermez.*")
st.caption("Geliştirenler: Emre Çelikkıran, Sinan Sukan, Yusuf Talha Akgül, Yasin Durmaz")
