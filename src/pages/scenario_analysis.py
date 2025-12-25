import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import folium
from streamlit_folium import folium_static

# Dosya yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCENARIOS_PATH = os.path.join(DATA_DIR, 'processed', 'scenarios.csv')
PREDICTIONS_PATH = os.path.join(DATA_DIR, 'processed', 'predictions.csv')
COORDINATES_PATH = os.path.join(DATA_DIR, 'maps', 'city_coordinates.json')
GEOJSON_PATH = os.path.join(DATA_DIR, 'maps', 'turkey_cities.geojson')

# Sayfa yapılandırması
st.set_page_config(page_title="Senaryo Analizi", page_icon="📊", layout="wide")

# Temel CSS stilleri
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        color: #1E3D59;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .section-divider {
        margin: 30px 0;
        border-bottom: 2px solid #eee;
    }
    /* Metric widget renkleri için */
    [data-testid="stMetricValue"] {
        color: inherit !important;
    }
    [data-testid="stMetricLabel"] {
        color: inherit !important;
    }
    [data-testid="stMetricDelta"] {
        color: inherit !important;
    }
    </style>
""", unsafe_allow_html=True)

# Parti renkleri ve stilleri
PARTI_RENKLERI = {
    'CHP': '#FF0000',      # Kırmızı
    'AK PARTİ': '#FFD700', # Sarı
    'MHP': '#0000FF',      # Mavi
    'HDP': '#800080'       # Mor
}

# Grafik teması
GRAFIK_TEMASI = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Arial, sans-serif'},
    'margin': dict(t=50, l=50, r=50, b=50)
}

# Veriyi oku
df = pd.read_csv(SCENARIOS_PATH)
df_tahmin = pd.read_csv(PREDICTIONS_PATH)

# Ana başlık
st.markdown('<p class="big-font">🗳️ 2024 Yerel Seçim Senaryoları</p>', unsafe_allow_html=True)

# Sidebar düzeni
with st.sidebar:
    st.markdown("### 📊 Senaryo Seçimi")
    selected_scenario = st.selectbox(
        "Analiz edilecek senaryoyu seçiniz:",
        df['Senaryo'].unique()
    )

# Senaryo verilerini filtrele
senaryo_df = df[df['Senaryo'] == selected_scenario].copy()

# Ana dashboard bölümü
st.markdown(f"### 📍 Seçili Senaryo: {selected_scenario}")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Türkiye geneli değişim istatistikleri
st.markdown("### 🇹🇷 Türkiye Geneli Değişim İstatistikleri")

# Senaryo için ağırlıklı ortalama hesaplama
total_votes_senaryo = senaryo_df['2024 CHP Tahmini Oy Sayısı'] + \
                      senaryo_df['2024 AK PARTİ Tahmini Oy Sayısı'] + \
                      senaryo_df['2024 MHP Tahmini Oy Sayısı'] + \
                      senaryo_df['2024 HDP Tahmini Oy Sayısı']

weighted_votes_senaryo = {
    "CHP": (senaryo_df['2024 CHP Tahmini Oy Sayısı'].sum() / total_votes_senaryo.sum()) * 100,
    "AK PARTİ": (senaryo_df['2024 AK PARTİ Tahmini Oy Sayısı'].sum() / total_votes_senaryo.sum()) * 100,
    "MHP": (senaryo_df['2024 MHP Tahmini Oy Sayısı'].sum() / total_votes_senaryo.sum()) * 100,
    "HDP": (senaryo_df['2024 HDP Tahmini Oy Sayısı'].sum() / total_votes_senaryo.sum()) * 100
}

# Tahmin için ağırlıklı ortalama hesaplama
total_votes_tahmin = df_tahmin['2024 CHP Tahmini Oy Sayısı'] + \
                     df_tahmin['2024 AK PARTİ Tahmini Oy Sayısı'] + \
                     df_tahmin['2024 MHP Tahmini Oy Sayısı'] + \
                     df_tahmin['2024 HDP Tahmini Oy Sayısı']

weighted_votes_tahmin = {
    "CHP": (df_tahmin['2024 CHP Tahmini Oy Sayısı'].sum() / total_votes_tahmin.sum()) * 100,
    "AK PARTİ": (df_tahmin['2024 AK PARTİ Tahmini Oy Sayısı'].sum() / total_votes_tahmin.sum()) * 100,
    "MHP": (df_tahmin['2024 MHP Tahmini Oy Sayısı'].sum() / total_votes_tahmin.sum()) * 100,
    "HDP": (df_tahmin['2024 HDP Tahmini Oy Sayısı'].sum() / total_votes_tahmin.sum()) * 100
}

col_stats = st.columns(4)
for idx, (col, parti) in enumerate(zip(col_stats, ['CHP', 'AK PARTİ', 'MHP', 'HDP'])):
    with col:
        # Senaryo ve tahmin değerlerini al
        senaryo_ort = weighted_votes_senaryo[parti]
        tahmin_ort = weighted_votes_tahmin[parti]
        
        # Farkı hesapla
        fark = senaryo_ort - tahmin_ort
        
        # Parti rengini al
        parti_rengi = PARTI_RENKLERI[parti]
        
        # Metin rengini belirle (açık renkler için koyu, koyu renkler için açık)
        text_color = '#FFFFFF' if parti in ['CHP', 'MHP', 'HDP'] else '#000000'
        
        st.markdown(f'<div style="background-color: {parti_rengi}; padding: 20px; border-radius: 10px; margin: 10px 0; color: {text_color};">', unsafe_allow_html=True)
        st.metric(
            label=f"{parti}",
            value=f"{senaryo_ort:.2f}%",
            delta=f"{fark:+.2f}%",
            delta_color="normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Harita görselleştirmesi - Senaryo sonuçlarına göre kazanan partiler
st.markdown("### 🗺️ Senaryo Haritası - Kazanan Partiler")

# Her ilin kazanan partisini belirle (senaryo için)
def get_scenario_winning_party(row):
    """Her il için senaryoda en yüksek oy oranına sahip partiyi bul"""
    parties = {
        "CHP": row['2024 CHP Tahmini Oy Sayısı (%)'],
        "AK PARTİ": row['2024 AK PARTİ Tahmini Oy Sayısı (%)'],
        "MHP": row['2024 MHP Tahmini Oy Sayısı (%)'],
        "HDP": row['2024 HDP Tahmini Oy Sayısı (%)']
    }
    return max(parties, key=parties.get)

# Senaryo kazanan partileri hesapla
senaryo_df['Senaryo Kazanan Parti'] = senaryo_df.apply(get_scenario_winning_party, axis=1)
senaryo_df['Senaryo Kazanan Oy Oranı'] = senaryo_df.apply(
    lambda row: row[f"2024 {row['Senaryo Kazanan Parti']} Tahmini Oy Sayısı (%)"], axis=1
)

# Mevcut kazanan partileri hesapla
def get_current_winning_party(row):
    """Her il için mevcut durumda en yüksek oy oranına sahip partiyi bul"""
    parties = {
        "CHP": row['CHP (%)'],
        "AK PARTİ": row['AK PARTİ (%)'],
        "MHP": row['MHP (%)'],
        "HDP": row['HDP (%)']
    }
    return max(parties, key=parties.get)

df_tahmin['Mevcut Kazanan Parti'] = df_tahmin.apply(get_current_winning_party, axis=1)
df_tahmin['Mevcut Kazanan Oy Oranı'] = df_tahmin.apply(
    lambda row: row[f"{row['Mevcut Kazanan Parti']} (%)"], axis=1
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

# İl adlarını normalize et
def normalize_city_name(name):
    """İl adını normalize et"""
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

# Senaryo verilerini harita için hazırla
scenario_city_mapping = {}
for idx, row in senaryo_df.iterrows():
    il_adi = row['İl Adı']
    current_row = df_tahmin[df_tahmin['İl Adı'] == il_adi].iloc[0] if len(df_tahmin[df_tahmin['İl Adı'] == il_adi]) > 0 else None
    
    scenario_city_mapping[il_adi] = {
        'scenario_winning_party': row['Senaryo Kazanan Parti'],
        'scenario_winning_percentage': row['Senaryo Kazanan Oy Oranı'],
        'current_winning_party': current_row['Mevcut Kazanan Parti'] if current_row is not None else 'Bilinmiyor',
        'current_winning_percentage': current_row['Mevcut Kazanan Oy Oranı'] if current_row is not None else 0,
        'chp_scenario': row['2024 CHP Tahmini Oy Sayısı (%)'],
        'akp_scenario': row['2024 AK PARTİ Tahmini Oy Sayısı (%)'],
        'mhp_scenario': row['2024 MHP Tahmini Oy Sayısı (%)'],
        'hdp_scenario': row['2024 HDP Tahmini Oy Sayısı (%)'],
        'chp_current': current_row['CHP (%)'] if current_row is not None else 0,
        'akp_current': current_row['AK PARTİ (%)'] if current_row is not None else 0,
        'mhp_current': current_row['MHP (%)'] if current_row is not None else 0,
        'hdp_current': current_row['HDP (%)'] if current_row is not None else 0
    }

# Harita oluştur
m_scenario = folium.Map(
    location=[39.0, 35.0],
    zoom_start=6,
    tiles='OpenStreetMap'
)

geojson_data = load_geojson()

if geojson_data:
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
    ).add_to(m_scenario)
else:
    # GeoJSON yoksa marker kullan
    @st.cache_data
    def load_coordinates():
        with open(COORDINATES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    coordinates = load_coordinates()
    
    for idx, row in senaryo_df.iterrows():
        il_adi = row['İl Adı']
        scenario_winning_party = row['Senaryo Kazanan Parti']
        scenario_winning_percentage = row['Senaryo Kazanan Oy Oranı']
        
        if il_adi in coordinates:
            lat = coordinates[il_adi]['lat']
            lon = coordinates[il_adi]['lon']
            color = PARTI_RENKLERI.get(scenario_winning_party, '#808080')
            
            city_data = scenario_city_mapping.get(il_adi, {})
            current_winning_party = city_data.get('current_winning_party', 'Bilinmiyor')
            changed = scenario_winning_party != current_winning_party
            change_indicator = "🔄" if changed else ""
            
            popup_html = f"""
            <div style="font-family: Arial; min-width: 280px;">
                <h4 style="margin: 5px 0; color: {color};">{il_adi} {change_indicator}</h4>
                <p style="margin: 5px 0;"><b>Senaryo Kazanan:</b> <span style="color: {color}; font-weight: bold;">{scenario_winning_party}</span> ({scenario_winning_percentage:.2f}%)</p>
                <p style="margin: 5px 0;"><b>Mevcut Kazanan:</b> {current_winning_party} ({city_data.get('current_winning_percentage', 0):.2f}%)</p>
                <hr style="margin: 8px 0;">
                <p style="margin: 5px 0; font-size: 11px;"><b>Senaryo Oy Oranları:</b></p>
                <p style="margin: 2px 0; font-size: 11px;">
                    CHP: {city_data.get('chp_scenario', 0):.2f}% | AK PARTİ: {city_data.get('akp_scenario', 0):.2f}%<br>
                    MHP: {city_data.get('mhp_scenario', 0):.2f}% | HDP: {city_data.get('hdp_scenario', 0):.2f}%
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
legend_html = '''
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
m_scenario.get_root().html.add_child(folium.Element(legend_html))

# Haritayı göster
folium_static(m_scenario, width=1200, height=600)

# Senaryo harita istatistikleri
col_scenario_map1, col_scenario_map2 = st.columns(2)

with col_scenario_map1:
    st.markdown("#### 📊 Parti Renkleri")
    st.markdown("""
    - 🔴 **CHP**: Kırmızı
    - 🟡 **AK PARTİ**: Sarı  
    - 🔵 **MHP**: Mavi
    - 🟣 **HDP**: Mor
    """)

with col_scenario_map2:
    st.markdown("#### 📈 Senaryo İstatistikleri")
    scenario_party_counts = senaryo_df['Senaryo Kazanan Parti'].value_counts()
    for party, count in scenario_party_counts.items():
        color_emoji = {"CHP": "🔴", "AK PARTİ": "🟡", "MHP": "🔵", "HDP": "🟣"}.get(party, "⚪")
        st.markdown(f"{color_emoji} **{party}**: {count} il")
    
    # Değişen iller sayısı
    merged_df = senaryo_df.merge(df_tahmin[['İl Adı', 'Mevcut Kazanan Parti']], on='İl Adı', how='left')
    changed_cities = (merged_df['Senaryo Kazanan Parti'] != merged_df['Mevcut Kazanan Parti']).sum()
    st.markdown(f"🔄 **Kazanan Parti Değişen İl Sayısı**: {changed_cities}")

st.caption(f"**{selected_scenario}** senaryosuna göre her il, en yüksek oy oranına sahip partiye göre renklendirilmiştir. "
           "🔄 işareti, kazanan partinin değiştiğini gösterir. İllere tıklayarak detaylı bilgi görebilirsiniz.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# İl bazlı analiz
st.markdown("### 🏛️ İl Bazlı Analiz")

# İl seçimi
selected_city = st.selectbox(
    "Detaylı analiz için il seçiniz:",
    sorted(senaryo_df['İl Adı'].unique())
)

# İl verilerini hazırla
il_data = []
for parti in ['CHP', 'AK PARTİ', 'MHP', 'HDP']:
    senaryo_col = f'2024 {parti} Tahmini Oy Sayısı (%)'
    tahmin_col = f'{parti} (%)'
    
    senaryo_deger = senaryo_df[senaryo_df['İl Adı'] == selected_city][senaryo_col].iloc[0]
    tahmin_deger = df_tahmin[df_tahmin['İl Adı'] == selected_city][tahmin_col].iloc[0]
    fark = senaryo_deger - tahmin_deger
    
    il_data.append({
        'Parti': parti,
        'Senaryo': senaryo_deger,
        'Tahmin': tahmin_deger,
        'Fark': fark
    })

il_df = pd.DataFrame(il_data)

# İl bazlı grafik
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Oy Oranları Karşılaştırması")
    fig = go.Figure()
    
    for parti in il_df['Parti']:
        parti_data = il_df[il_df['Parti'] == parti]
        
        fig.add_trace(go.Bar(
            name=f"{parti} - Senaryo",
            x=[parti],
            y=[parti_data['Senaryo'].iloc[0]],
            marker_color=PARTI_RENKLERI[parti],
            width=0.3,
            offset=-0.2
        ))
        
        fig.add_trace(go.Bar(
            name=f"{parti} - Tahmin",
            x=[parti],
            y=[parti_data['Tahmin'].iloc[0]],
            marker_color=PARTI_RENKLERI[parti],
            opacity=0.5,
            width=0.3,
            offset=0.2
        ))
    
    fig.update_layout(
        **GRAFIK_TEMASI,
        title=f"{selected_city} - Senaryo vs Tahmin",
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Geliştiriciler
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Geliştirenler: Emre Çelikkıran, Sinan Sukan, Yusuf Talha Akgül, Yasin Durmaz")

with col2:
    st.markdown("#### 📈 Değişim Analizi")
    for _, row in il_df.iterrows():
        # Seçili il için ağırlıklı ortalama değerleri al
        parti = row['Parti']
        senaryo_col = f'2024 {parti} Tahmini Oy Sayısı (%)'
        tahmin_col = f'{parti} (%)'
        
        # Seçili il için değerleri al
        senaryo_deger = senaryo_df[senaryo_df['İl Adı'] == selected_city][senaryo_col].iloc[0]
        tahmin_deger = df_tahmin[df_tahmin['İl Adı'] == selected_city][tahmin_col].iloc[0]
        fark = senaryo_deger - tahmin_deger
        
        if fark > 0:
            emoji = "📈"
            color = "#28a745"
        else:
            emoji = "📉"
            color = "#dc3545"
            
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="color: {color}; margin: 0;">{parti} {emoji}</h4>
            <p style="margin: 5px 0;">
                Senaryo: <b>{senaryo_deger:.2f}%</b><br>
                Tahmin: <b>{tahmin_deger:.2f}%</b><br>
                Değişim: <b>{fark:+.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Tüm iller karşılaştırması
st.markdown("### 🗺️ Tüm İller Karşılaştırması")

# İl seçimi
selected_cities = st.multiselect(
    "Karşılaştırma için il seçiniz:",
    sorted(senaryo_df['İl Adı'].unique()),
    default=sorted(senaryo_df['İl Adı'].unique())[:5]
)

if selected_cities:
    filtered_df = senaryo_df[senaryo_df['İl Adı'].isin(selected_cities)]
    
    fig = go.Figure()
    
    for parti in ['CHP', 'AK PARTİ', 'MHP', 'HDP']:
        senaryo_col = f'2024 {parti} Tahmini Oy Sayısı (%)'
        
        fig.add_trace(go.Bar(
            name=parti,
            x=filtered_df['İl Adı'],
            y=filtered_df[senaryo_col],
            marker_color=PARTI_RENKLERI[parti]
        ))
    
    fig.update_layout(
        **GRAFIK_TEMASI,
        title="İllere Göre Parti Oy Oranları",
        barmode='group',
        height=500,
        xaxis_title="İller",
        yaxis_title="Oy Oranı (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

