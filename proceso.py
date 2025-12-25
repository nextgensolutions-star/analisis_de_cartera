import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import io
import google.generativeai as genai

# Configuración de la página
st.set_page_config(page_title="Terminal Económica + IA", layout="wide")
st.title("📊 Terminal Económica con Asistente IA")

# --- CONFIGURACIÓN DE IA ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.warning("Configura GEMINI_API_KEY en los secretos de Streamlit para activar el chat.")

# --- INICIALIZAR ESTADO DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCIONES DE AYUDA ---
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Sheet1')
    writer.close()
    return output.getvalue()

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acciones", "Criptos", "Forex", "Índices"])
ticker_map = {"Acciones": "AAPL", "Criptos": "BTC-USD", "Forex": "EURUSD=X", "Índices": "^GSPC"}
ticker = st.sidebar.text_input("Ticker", ticker_map[tipo_activo])

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    fecha_inicio = st.date_input("Desde", datetime.now() - timedelta(days=365*2))
with col_f2:
    fecha_fin = st.date_input("Hasta", datetime.now())

frecuencia_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
mapa_resample = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "YE"}

opcion = st.sidebar.selectbox(
    "Selecciona el Análisis",
    ["Precios y Retornos", "Análisis Econométrico", "Datos Fundamentales", "🤖 Asistente IA"]
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
    retornos_all = np.log(data / data.shift(1)).dropna()
    
    # --- LÓGICA DE INTERFAZ ---
    if opcion == "Precios y Retornos":
        rendimiento_total = (asset_series.iloc[-1] / asset_series.iloc[0]) - 1
        m1, m2 = st.columns(2)
        m1.metric("Rendimiento Agregado", f"{rendimiento_total:.2%}")
        m2.metric("Precio Final", f"{asset_series.iloc[-1]:.2f}")
        
        tab_lin, tab_dist = st.tabs(["Gráficos Lineales", "Distribución"])
        with tab_lin:
            st.plotly_chart(px.line(asset_series, title="Precio"), use_container_width=True)
            st.plotly_chart(px.line(retornos_all[ticker], title="Retornos Log"), use_container_width=True)
        with tab_dist:
            st.plotly_chart(px.histogram(retornos_all[ticker], title="Histograma de Retornos", marginal="box"), use_container_width=True)

    elif opcion == "Análisis Econométrico":
        st.subheader("Modelos")
        Y = retornos_all[ticker]
        X = sm.add_constant(retornos_all["SPY"])
        modelo = sm.OLS(Y, X).fit()
        res_adf = adfuller(asset_series)
        
        st.write(f"**Beta vs SP500:** {modelo.params[1]:.4f}")
        st.write(f"**P-Value ADF:** {res_adf[1]:.4f}")
        st.text(modelo.summary())

    elif opcion == "Datos Fundamentales":
        empresa = yf.Ticker(ticker)
        st.dataframe(empresa.balance_sheet)

    elif opcion == "🤖 Asistente IA":
        st.subheader("Consultorio Econométrico Virtual")
        st.write("Pregúntale a la IA sobre los resultados obtenidos (Beta, estacionariedad, etc.)")
        
        # Mostramos historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("¿Qué significa que mi Beta sea mayor a 1?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Preparar contexto para la IA
            contexto = f"""
            El usuario está analizando el ticker {ticker}. 
            Frecuencia: {frecuencia_label}.
            Resultados actuales:
            - Beta vs SP500: {sm.OLS(retornos_all[ticker], sm.add_constant(retornos_all["SPY"])).fit().params[1]:.4f}
            - P-value ADF: {adfuller(asset_series)[1]:.4f}
            Pregunta del usuario: {prompt}
            """
            
            with st.chat_message("assistant"):
                response = model.generate_content(contexto)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

    # --- EXPORTACIÓN ---
    st.sidebar.markdown("---")
    st.sidebar.download_button("Descargar CSV", data=data.to_csv().encode('utf-8'), file_name=f"{ticker}.csv")
    st.sidebar.download_button("Descargar Excel", data=to_excel(data), file_name=f"{ticker}.xlsx")

except Exception as e:
    st.error(f"Error: {e}")