import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
import io

st.set_page_config(page_title="Prognoza sprzedaży", layout="centered")
st.title("📈 Prognoza sprzedaży")

DEFAULT_CSV_PATH = "data/2023_sprzedaz_b2b.csv"
uploaded_file = st.file_uploader("📂 Wgraj plik CSV z danymi sprzedaży (lub użyj domyślnego)")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
else:
    df = pd.read_csv(DEFAULT_CSV_PATH, sep=';')
    st.info("📌 Użyto domyślnej bazy danych.")

# Czyszczenie danych
df['ilosc'] = pd.to_numeric(df['ilosc'], errors='coerce')
df['wartosc_netto_pln'] = (
    df['wartosc_netto_pln']
    .astype(str)
    .str.replace(u'\xa0', '', regex=True)
    .str.replace(' ', '', regex=False)
    .str.replace(',', '.', regex=False)
)
df['wartosc_netto_pln'] = pd.to_numeric(df['wartosc_netto_pln'], errors='coerce')
df = df[(df['ilosc'] > 0) & (df['wartosc_netto_pln'] > 0)]

# Przeliczenia
df['cena_jednostkowa'] = df['wartosc_netto_pln'] / df['ilosc']
df['wartosc_netto_pln'] = (df['ilosc'] * df['cena_jednostkowa']).round(2)
df['ilosc'] = df['ilosc'].round(0).astype('Int64')
df['data'] = pd.to_datetime(
    df['Rok_data_sprzedazy'].astype(str) + '-' +
    df['Miesiac_data_sprzedazy'].astype(str).str.zfill(2) + '-01'
)

# Tryb działania
tryb_prognozy = st.radio("📌 Tryb prognozy:", [
    "Zbiorcza tabela",
    "Szczegółowa (wykresy per SKU)"
])

st.subheader("🎛️ Filtry danych")

# Kategorie
wszystkie_kategorie = sorted(df['Kategoria_Produktu'].unique())
zaznacz_kategorie = st.checkbox("✅ Zaznacz wszystkie kategorie", value=True)
with st.expander("🏷️ Wybierz kategorie produktów", expanded=False):
    wybrane_kategorie = st.multiselect(
        "Kategorie:", wszystkie_kategorie,
        default=wszystkie_kategorie if zaznacz_kategorie else []
    )
df_filtered = df[df['Kategoria_Produktu'].isin(wybrane_kategorie)]

# SKU
wszystkie_sku = sorted(df_filtered['sku'].unique())
zaznacz_sku = st.checkbox("✅ Zaznacz wszystkie SKU", value=True)
with st.expander("📦 Wybierz produkty (SKU)", expanded=False):
    wybrane_sku = st.multiselect(
        "Produkty (SKU):", wszystkie_sku,
        default=wszystkie_sku if zaznacz_sku else []
    )
df_filtered = df_filtered[df_filtered['sku'].isin(wybrane_sku)]

agregat = st.radio("📊 Prognozuj według:", ['ilosc', 'wartosc_netto_pln'])
miesiace = st.slider("📅 Na ile miesięcy prognoza?", 1, 12, 3)

# 🔁 ZBIORCZA TABELA
if tryb_prognozy == "Zbiorcza tabela":
    podtryb = st.radio("📋 Co chcesz zobaczyć?", [
        "Suma prognozy per SKU",
        "Prognoza miesięczna per SKU"
    ])
    tabela_sumaryczna = []

    for sku in wybrane_sku:
        df_sku = df_filtered[df_filtered['sku'] == sku].copy()
        df_agg = df_sku.groupby('data')[agregat].sum().reset_index()
        df_agg.columns = ['ds', 'y']
        df_agg['y'] = pd.to_numeric(df_agg['y'], errors='coerce')

        if df_agg['y'].notna().sum() < 2:
            continue

        model = Prophet()
        model.fit(df_agg)

        last_date = df_agg['ds'].max()
        next_year = last_date.year + 1
        future_dates = pd.date_range(start=f"{next_year}-01-01", periods=miesiace, freq='MS')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

        if podtryb == "Suma prognozy per SKU":
            suma = forecast[['yhat', 'yhat_lower', 'yhat_upper']].sum()
            tabela_sumaryczna.append({
                'SKU': sku,
                'Prognoza': suma['yhat'],
                'Min': suma['yhat_lower'],
                'Max': suma['yhat_upper']
            })
        else:
            prognoza_mies = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            prognoza_mies['SKU'] = sku
            prognoza_mies.columns = ['Miesiąc', 'Prognoza', 'Min', 'Max', 'SKU']
            prognoza_mies = prognoza_mies[['SKU', 'Miesiąc', 'Prognoza', 'Min', 'Max']]
            tabela_sumaryczna.append(prognoza_mies)

    if podtryb == "Suma prognozy per SKU":
        df_wynik = pd.DataFrame(tabela_sumaryczna)

        sortuj_po_nabywcy = st.checkbox("🔠 Sortuj po nabywcy (jeśli dotyczy)", value=False)
        if sortuj_po_nabywcy and 'nabywca' in df_wynik.columns:
            df_wynik = df_wynik.sort_values("nabywca")

        suma_all = df_wynik[['Prognoza', 'Min', 'Max']].sum()
        suma_row = pd.DataFrame([{
            'SKU': 'SUMA',
            'Prognoza': suma_all['Prognoza'],
            'Min': suma_all['Min'],
            'Max': suma_all['Max']
        }])
        df_wynik = pd.concat([df_wynik, suma_row], ignore_index=True)

        if agregat == 'ilosc':
            df_wynik[['Prognoza', 'Min', 'Max']] = df_wynik[['Prognoza', 'Min', 'Max']].round(0).astype('Int64')
            fmt = '{:,.0f}'
        else:
            df_wynik[['Prognoza', 'Min', 'Max']] = df_wynik[['Prognoza', 'Min', 'Max']].round(2)
            fmt = '{:,.2f}'

        def highlight_suma(row):
            return ['background-color: #f0f0f0; font-weight: bold' if row['SKU'] == 'SUMA' else '' for _ in row]

        st.dataframe(df_wynik.style.apply(highlight_suma, axis=1).format({
            'Prognoza': fmt, 'Min': fmt, 'Max': fmt
        }))

        # 📤 Eksport do Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_wynik.to_excel(writer, sheet_name='Prognoza', index=False)
            writer.close()

        st.download_button(
            label="📥 Pobierz prognozę jako Excel",
            data=buffer,
            file_name="prognoza_sprzedazy.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        df_prognoza = pd.concat(tabela_sumaryczna, ignore_index=True)
        df_prognoza[['Prognoza', 'Min', 'Max']] = df_prognoza[['Prognoza', 'Min', 'Max']].round(2)
        st.dataframe(df_prognoza)

# 📊 Szczegółowe wykresy
elif tryb_prognozy == "Szczegółowa (wykresy per SKU)":
    st.subheader("📉 Prognozy szczegółowe")

    for sku in wybrane_sku:
        df_sku = df_filtered[df_filtered['sku'] == sku].copy()
        df_agg = df_sku.groupby('data')[agregat].sum().reset_index()
        df_agg.columns = ['ds', 'y']
        df_agg['y'] = pd.to_numeric(df_agg['y'], errors='coerce')

        if df_agg['y'].notna().sum() < 2:
            st.warning(f"⚠️ Za mało danych dla SKU: {sku}")
            continue

        model = Prophet()
        model.fit(df_agg)

        last_date = df_agg['ds'].max()
        next_year = last_date.year + 1
        future_dates = pd.date_range(start=f"{next_year}-01-01", periods=miesiace, freq='MS')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

        st.markdown(f"### 📦 Prognoza dla SKU: `{sku}`")
        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)
        st.pyplot(fig)

        tabela = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        tabela = tabela.rename(columns={
            'ds': 'Miesiąc',
            'yhat': 'Prognoza',
            'yhat_lower': 'Min',
            'yhat_upper': 'Max'
        })

        if agregat == 'ilosc':
            tabela[['Prognoza', 'Min', 'Max']] = tabela[['Prognoza', 'Min', 'Max']].round(0).astype('Int64')
        else:
            tabela[['Prognoza', 'Min', 'Max']] = tabela[['Prognoza', 'Min', 'Max']].round(2)

        st.dataframe(tabela)
