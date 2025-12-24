import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import io

# Configuración de la página
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")
st.title("📊 Terminal de Análisis Económico y Financiero")

# --- FUNCIONES DE AYUDA ---
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Sheet1')
    writer.close()
    return output.getvalue()

# --- BARRA LATERAL ---
st.sidebar.header("Configuración de Datos")

tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acciones", "Criptos", "Forex", "Índices"])
ticker_map = {"Acciones": "AAPL", "Criptos": "BTC-USD", "Forex": "EURUSD=X", "Índices": "^GSPC"}
ticker = st.sidebar.text_input("Ticker", ticker_map[tipo_activo])

col1, col2 = st.sidebar.columns(2)
with col1:
    fecha_inicio = st.date_input("Desde", datetime.now() - timedelta(days=365*2))
with col2:
    fecha_fin = st.date_input("Hasta", datetime.now())

frecuencia_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
mapa_resample = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "YE"}

opcion = st.sidebar.selectbox(
    "Selecciona el Análisis",
    ["Precios y Retornos", "Análisis Econométrico", "Datos Fundamentales", "Perfil de Empresa"]
)

# --- PROCESAMIENTO ---
@st.cache_data
def obtener_datos(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end)['Close']
    df_res = df.resample(mapa_resample[freq]).last().dropna()
    return df_res

try:
    data = obtener_datos(ticker, fecha_inicio, fecha_fin, frecuencia_label)
    asset_series = data[ticker]
    
    # --- CONTENIDO PRINCIPAL ---
    if opcion == "Precios y Retornos":
        st.subheader(f"Análisis de {ticker} ({frecuencia_label})")
        
        # 1. Cálculos de Rendimiento
        retornos = np.log(asset_series / asset_series.shift(1)).dropna()
        rendimiento_total = (asset_series.iloc[-1] / asset_series.iloc[0]) - 1
        volatilidad = retornos.std() * np.sqrt(252) if frecuencia_label == "Diario" else retornos.std()

        # 2. Tarjetas de Métricas (Key Metrics)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rendimiento Agregado", f"{rendimiento_total:.2%}")
        m2.metric("Precio Final", f"{asset_series.iloc[-1]:.2f}")
        m3.metric("Volatilidad (Std Dev)", f"{volatilidad:.4f}")

        # 3. Gráficos
        tab1, tab2 = st.tabs(["Series Temporales", "Distribución de Retornos"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(asset_series, title="Evolución de Precios", labels={'value': 'Precio', 'Date': 'Fecha'}), use_container_width=True)
            with c2:
                st.plotly_chart(px.line(retornos, title="Retornos Logarítmicos", labels={'value': 'Retorno Log', 'Date': 'Fecha'}), use_container_width=True)
        
        with tab2:
            st.write("El histograma muestra qué tan frecuentes son las subas y bajas. Una campana ancha indica mayor riesgo.")
            fig_hist = px.histogram(retornos, 
                                    nbins=50, 
                                    title="Distribución de Retornos (Histograma)",
                                    labels={'value': 'Retorno'},
                                    marginal="box") # Agrega un diagrama de caja arriba para ver outliers
            st.plotly_chart(fig_hist, use_container_width=True)

    elif opcion == "Análisis Econométrico":
        st.subheader(f"Modelos sobre datos {frecuencia_label}")
        retornos_all = np.log(data / data.shift(1)).dropna()
        
        t1, t2 = st.tabs(["Regresión OLS", "Test de Estacionariedad"])
        with t1:
            Y = retornos_all[ticker]
            X = sm.add_constant(retornos_all["SPY"])
            modelo = sm.OLS(Y, X).fit()
            st.write(f"**Beta ({frecuencia_label}):** {modelo.params[1]:.4f}")
            st.text(modelo.summary())
        with t2:
            res_adf = adfuller(asset_series)
            st.metric("p-value ADF", f"{res_adf[1]:.4f}")
            if res_adf[1] < 0.05:
                st.success("Serie Estacionaria")
            else:
                st.warning("Serie No Estacionaria")

    elif opcion == "Datos Fundamentales":
        st.subheader(f"Estados Financieros - {ticker}")
        empresa = yf.Ticker(ticker)
        tipo_estado = st.radio("Reporte:", ["Balance Sheet", "Income Statement", "Cash Flow"], horizontal=True)
        
        df_fund = empresa.balance_sheet if tipo_estado == "Balance Sheet" else (empresa.income_stmt if tipo_estado == "Income Statement" else empresa.cashflow)
        st.dataframe(df_fund)

    # --- EXPORTACIÓN (Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Exportar")
    csv_data = data.to_csv().encode('utf-8')
    st.sidebar.download_button("Descargar CSV", data=csv_data, file_name=f"{ticker}_datos.csv")
    st.sidebar.download_button("Descargar Excel", data=to_excel(data), file_name=f"{ticker}_datos.xlsx")

except Exception as e:
    st.error(f"Error: {e}. Verifica el Ticker o el rango de fechas.")