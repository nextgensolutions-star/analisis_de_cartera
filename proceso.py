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

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")
st.title("📊 Terminal de Análisis Económico y Financiero")

# --- 2. CONFIGURACIÓN DE IA (Instrucciones del Sistema) ---
instrucciones_ia = """
Eres un Asistente Experto en Econometría de la Maestría en Economía Aplicada. 
Tu función es interpretar resultados de modelos financieros (Beta, OLS, ADF) 
de forma académica y clara. Cita conceptos técnicos cuando sea necesario.
"""

# Inicializar IA de forma segura
ia_lista = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model_ia = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=instrucciones_ia
        )
        ia_lista = True
    else:
        st.sidebar.warning("⚠️ Chatbot: Falta GEMINI_API_KEY en los secretos.")
except Exception as e:
    st.sidebar.error(f"Error al configurar IA: {e}")

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. BARRA LATERAL (FILTROS) ---
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

# ESTA ES LA CLAVE: El menú de navegación
opcion = st.sidebar.selectbox(
    "Selecciona el Análisis",
    ["Precios y Retornos", "Análisis Econométrico", "Datos Fundamentales", "🤖 Asistente IA"]
)

# --- 4. PROCESAMIENTO DE DATOS ---
@st.cache_data
def obtener_datos(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end)['Close']
    df_res = df.resample(mapa_resample[freq]).last().dropna()
    return df_res

try:
    data = obtener_datos(ticker, fecha_inicio, fecha_fin, frecuencia_label)
    asset_series = data[ticker]
    retornos_all = np.log(data / data.shift(1)).dropna()

    # --- 5. LÓGICA DE NAVEGACIÓN (Lo que aparece en pantalla) ---
    
    if opcion == "Precios y Retornos":
        rendimiento_total = (asset_series.iloc[-1] / asset_series.iloc[0]) - 1
        st.metric("Rendimiento Agregado", f"{rendimiento_total:.2%}")
        
        tab_lin, tab_dist = st.tabs(["Gráficos", "Distribución"])
        with tab_lin:
            st.plotly_chart(px.line(asset_series, title="Precio"), use_container_width=True)
            st.plotly_chart(px.line(retornos_all[ticker], title="Retornos"), use_container_width=True)
        with tab_dist:
            st.plotly_chart(px.histogram(retornos_all[ticker], title="Histograma", marginal="box"), use_container_width=True)

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
        st.dataframe(yf.Ticker(ticker).balance_sheet)

    elif opcion == "🤖 Asistente IA":
        st.subheader("🤖 Consultor Econométrico Virtual")
        
        if not ia_lista:
            st.error("El chatbot no está configurado correctamente. Revisa tu GEMINI_API_KEY.")
        else:
            # Mostrar mensajes previos
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Entrada de usuario
            if prompt := st.chat_input("¿Qué significan estos resultados?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Contexto dinámico
                beta = sm.OLS(retornos_all[ticker], sm.add_constant(retornos_all["SPY"])).fit().params[1]
                adf_p = adfuller(asset_series)[1]
                
                contexto = f"Ticker: {ticker}. Beta: {beta:.4f}. ADF p-value: {adf_p:.4f}. Pregunta: {prompt}"
                
                with st.chat_message("assistant"):
                    response = model_ia.generate_content(contexto)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

except Exception as e:
    st.error(f"Error en la aplicación: {e}")