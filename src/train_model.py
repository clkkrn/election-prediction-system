import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Dosya yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'election_data.xlsx')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Veri dosyasını yükleyin
excel_file_path = RAW_DATA_PATH
data = pd.ExcelFile(excel_file_path)
df = data.parse('Sheet1')

# Bağımlı ve bağımsız değişkenleri tanımlayalım
dependent_variables = df[['2024 CHP Oy Sayısı', '2024 AK PARTİ Oy Sayısı', '2024 Son MHP Oy Sayısı', '2024 HDP Parti Oy Sayısı']]
independent_variables = df.drop(columns=['2024 CHP Oy Sayısı', '2024 AK PARTİ Oy Sayısı', '2024 Son MHP Oy Sayısı', '2024 HDP Parti Oy Sayısı'])

# Kategorik değişkenleri encode edelim (One-hot Encoding)
categorical_columns = independent_variables.select_dtypes(include=['object']).columns
independent_variables = pd.get_dummies(independent_variables, columns=categorical_columns, drop_first=True)

# Önemli özellikleri belirleyelim
important_features = [
    'Seçmen Sayısı', 'Kadın Seçmen Sayısı', 'Erkek Seçmen Sayısı',
    'Kişi Başına Düşen Gelir', 'Eğitim Düzeyi (Lise+)', '65+ Yaşlı Nüfus (Emekli)',
    '2009 AK PARTİ Oy Sayısı','2009 CHP Oy Sayısı','2009 MHP Oy Sayısı','2009 HDP Parti Oy Sayısı', '2014 AK PARTİ Oy Sayısı','2014 CHP Oy Sayısı','2014 MHP Oy Sayısı', '2014 HDP Parti Oy Sayısı', '2019 AK PARTİ Oy Sayısı','2019 CHP Oy Sayısı','2019 MHP Oy Sayısı','2019 HDP Parti Oy Sayısı',
    '2024 Bağımsız Oy Sayısı', '2024 Büyük Birlik Partisi Oy Sayısı',
    'Anket Örneklemi', 'CHP Anket Oy Oranı', 'İYİ Parti Anket Oy Oranı',
    'HDP Parti Anket Oy Oranı', 'MHP Anket Oy Oranı',
    'İl Adı_ANKARA', 'İl Adı_İSTANBUL', 'İl Adı_İZMİR',
    '2019 Kazanan Parti_CHP', '2024 Kazanan Parti_CHP', 'AKP Anket Oy Oranı', 'İşsizlik Oranı'
]
X_selected = independent_variables[important_features]

# Eksik değerleri doldur
X_selected = X_selected.fillna(X_selected.mean())

# Eğitim ve test setlerini oluştur
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected, dependent_variables, test_size=0.2, random_state=42
)

# XGBoost Modeli
xgb_model_chp = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, eval_metric='rmse')
xgb_model_chp.fit(X_train_selected, y_train_selected['2024 CHP Oy Sayısı'])
y_pred_chp = xgb_model_chp.predict(X_test_selected)
# XGBoost 3.x uyumluluğu için pickle kullanıyoruz
with open(os.path.join(MODELS_DIR, "chp_model.json"), 'wb') as f:
    pickle.dump(xgb_model_chp, f)

xgb_model_akp = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, eval_metric='rmse')
xgb_model_akp.fit(X_train_selected, y_train_selected['2024 AK PARTİ Oy Sayısı'])
y_pred_akp = xgb_model_akp.predict(X_test_selected)
with open(os.path.join(MODELS_DIR, "akp_model.json"), 'wb') as f:
    pickle.dump(xgb_model_akp, f)

xgb_model_mhp = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, eval_metric='rmse')
xgb_model_mhp.fit(X_train_selected, y_train_selected['2024 Son MHP Oy Sayısı'])
y_pred_mhp = xgb_model_mhp.predict(X_test_selected)
with open(os.path.join(MODELS_DIR, "mhp_model.json"), 'wb') as f:
    pickle.dump(xgb_model_mhp, f)

xgb_model_hdp = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, eval_metric='rmse')
xgb_model_hdp.fit(X_train_selected, y_train_selected['2024 HDP Parti Oy Sayısı'])
y_pred_hdp = xgb_model_hdp.predict(X_test_selected)
with open(os.path.join(MODELS_DIR, "hdp_model.json"), 'wb') as f:
    pickle.dump(xgb_model_hdp, f)

# Performans değerlendirme: R2 ve MSE
r2_chp = r2_score(y_test_selected['2024 CHP Oy Sayısı'], y_pred_chp)
mse_chp = mean_squared_error(y_test_selected['2024 CHP Oy Sayısı'], y_pred_chp)

r2_akp = r2_score(y_test_selected['2024 AK PARTİ Oy Sayısı'], y_pred_akp)
mse_akp = mean_squared_error(y_test_selected['2024 AK PARTİ Oy Sayısı'], y_pred_akp)

r2_mhp = r2_score(y_test_selected['2024 Son MHP Oy Sayısı'], y_pred_mhp)
mse_mhp = mean_squared_error(y_test_selected['2024 Son MHP Oy Sayısı'], y_pred_mhp)

r2_hdp = r2_score(y_test_selected['2024 HDP Parti Oy Sayısı'], y_pred_hdp)
mse_hdp = mean_squared_error(y_test_selected['2024 HDP Parti Oy Sayısı'], y_pred_hdp)

print(f"CHP Oy Tahmini: R2={r2_chp}, MSE={mse_chp}")
print(f"AK PARTİ Oy Tahmini: R2={r2_akp}, MSE={mse_akp}")
print(f"MHP Oy Tahmini: R2={r2_mhp}, MSE={mse_mhp}")
print(f"HDP Oy Tahmini: R2={r2_hdp}, MSE={mse_hdp}")

# 81 il için tahmin sonuçlarını oluştur
predictions_chp = xgb_model_chp.predict(X_selected)
predictions_akp = xgb_model_akp.predict(X_selected)
predictions_mhp = xgb_model_mhp.predict(X_selected)
predictions_hdp = xgb_model_hdp.predict(X_selected)

predictions_df = pd.DataFrame({
    'İl Adı': df['İl Adı'],
    '2024 CHP Tahmini Oy Sayısı': predictions_chp,
    '2024 AK PARTİ Tahmini Oy Sayısı': predictions_akp,
    '2024 MHP Tahmini Oy Sayısı': predictions_mhp,
    '2024 HDP Tahmini Oy Sayısı': predictions_hdp
})

# Parti yüzdelerini hesaplayın
predictions_df['CHP (%)'] = (predictions_df['2024 CHP Tahmini Oy Sayısı'] / (
    predictions_df['2024 CHP Tahmini Oy Sayısı'] + predictions_df['2024 AK PARTİ Tahmini Oy Sayısı'] + predictions_df['2024 MHP Tahmini Oy Sayısı'] + predictions_df['2024 HDP Tahmini Oy Sayısı'])) * 100
predictions_df['AK PARTİ (%)'] = (predictions_df['2024 AK PARTİ Tahmini Oy Sayısı'] / (
    predictions_df['2024 CHP Tahmini Oy Sayısı'] + predictions_df['2024 AK PARTİ Tahmini Oy Sayısı'] + predictions_df['2024 MHP Tahmini Oy Sayısı'] + predictions_df['2024 HDP Tahmini Oy Sayısı'])) * 100
predictions_df['MHP (%)'] = (predictions_df['2024 MHP Tahmini Oy Sayısı'] / (
    predictions_df['2024 CHP Tahmini Oy Sayısı'] + predictions_df['2024 AK PARTİ Tahmini Oy Sayısı'] + predictions_df['2024 MHP Tahmini Oy Sayısı'] + predictions_df['2024 HDP Tahmini Oy Sayısı'])) * 100
predictions_df['HDP (%)'] = (predictions_df['2024 HDP Tahmini Oy Sayısı'] / (
    predictions_df['2024 CHP Tahmini Oy Sayısı'] + predictions_df['2024 AK PARTİ Tahmini Oy Sayısı'] + predictions_df['2024 MHP Tahmini Oy Sayısı'] + predictions_df['2024 HDP Tahmini Oy Sayısı'])) * 100

# Tahmin sonuçlarını CSV olarak kaydedin
output_csv_path = os.path.join(PROCESSED_DIR, 'predictions.csv')
predictions_df.to_csv(output_csv_path, index=False)

print(f"Tahmin sonuçları başarıyla {output_csv_path} dosyasına kaydedildi.")

models = {
    '2024 CHP Tahmini Oy Sayısı': xgb_model_chp,
    '2024 AK PARTİ Tahmini Oy Sayısı': xgb_model_akp,
    '2024 MHP Tahmini Oy Sayısı': xgb_model_mhp,
    '2024 HDP Tahmini Oy Sayısı': xgb_model_hdp
}


def predict_with_scenario(model, base_X, scenario_changes):
    scenario_X = base_X.copy()
    for feature, change in scenario_changes.items():
        if feature in scenario_X.columns:
            scenario_X[feature] *= change
    predictions = model.predict(scenario_X)
    return predictions

# Mevcut senaryo analiz fonksiyonunu yeniden kullanarak birden fazla senaryoyu işleme
def collect_scenarios(base_X, models, scenarios):
    all_scenario_results = []

    for scenario_name, changes in scenarios.items():
        scenario_results = {column: [] for column in models.keys()}
        for _, row in base_X.iterrows():
            row_numeric = row.to_frame().T.apply(pd.to_numeric, errors='coerce')
            for column, model in models.items():
                prediction = predict_with_scenario(model, row_numeric, changes)
                scenario_results[column].append(prediction[0])

        # Sonuçları bir DataFrame'e dönüştür ve senaryo adı ekle
        scenario_df = pd.DataFrame({
            'İl Adı': df['İl Adı'],
            **scenario_results
        })
        scenario_df['Senaryo'] = scenario_name

        # Parti yüzdelerini hesapla
        scenario_df['Toplam Oy'] = scenario_df[list(models.keys())].sum(axis=1)
        for col in models.keys():
            scenario_df[f'{col} (%)'] = (scenario_df[col] / scenario_df['Toplam Oy']) * 100

        all_scenario_results.append(scenario_df)

    # Tüm senaryoları birleştir
    combined_df = pd.concat(all_scenario_results, ignore_index=True)
    return combined_df

# Tanımlı senaryolar
scenarios = {
    "Gelir +%20": {'Kişi Başına Düşen Gelir': 1.20},
    "Gelir -%20": {'Kişi Başına Düşen Gelir': 0.80},
    "Emekli Nüfus +%20": {'65+ Yaşlı Nüfus (Emekli)': 1.20},
    "Emekli Nüfus -%20": {'65+ Yaşlı Nüfus (Emekli)': 0.80},
    "Eğitim Düzeyi +%20": {'Eğitim Düzeyi (Lise+)': 1.20},
    "Eğitim Düzeyi -%20": {'Eğitim Düzeyi (Lise+)': 0.80},
    "işsizlik Oranı +%20": {'İşsizlik Oranı': 1.20},
    "İşsizlik Oranı -%20": {'İşsizlik Oranı': 0.80},
    "CHP Anket Artışı +%20": {'CHP Anket Oy Oranı': 1.20},
    "CHP Anket Artışı -%20": {'CHP Anket Oy Oranı': 0.80},
    "AKP Anket Artışı +%20": {'AKP Anket Oy Oranı': 1.20},
    "AKP Anket Artışı -%20": {'AKP Anket Oy Oranı': 0.80},

}

# Tüm senaryoları çalıştır ve sonuçları kaydet
combined_scenarios = collect_scenarios(X_selected, models, scenarios)
output_scenario_path = os.path.join(PROCESSED_DIR, 'scenarios.csv')
combined_scenarios.to_csv(output_scenario_path, index=False)

print(f"Tüm senaryo tahmin sonuçları başarıyla {output_scenario_path} dosyasına kaydedildi.")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# CHP, AK PARTİ, MHP, HDP için doğruluk grafikleri
parties = ['2024 CHP Oy Sayısı', '2024 AK PARTİ Oy Sayısı', '2024 Son MHP Oy Sayısı', '2024 HDP Parti Oy Sayısı']
predictions = [y_pred_chp, y_pred_akp, y_pred_mhp, y_pred_hdp]
real_values = [y_test_selected[col] for col in parties]
metrics = [r2_chp, r2_akp, r2_mhp, r2_hdp]

for i, (party, y_pred, y_real, r2) in enumerate(zip(parties, predictions, real_values, metrics)):
    axes[i].scatter(y_real, y_pred, alpha=0.7, label=f"R²: {r2:.2f}")
    axes[i].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
    axes[i].set_title(f"{party} Tahmin Sonuçları")
    axes[i].set_xlabel("Gerçek Değerler")
    axes[i].set_ylabel("Tahmin Değerleri")
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "model_accuracy_plots.png"))
print(f"Model doğruluk grafikleri {os.path.join(BASE_DIR, 'model_accuracy_plots.png')} dosyasına kaydedildi.")
plt.show()

