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

# 1. CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")

# Estilo para el botón del chat
st.markdown("""
    <style>
    .stPopover { position: fixed; bottom: 20px; right: 20px; z-index: 1000; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Terminal de Análisis Económico")

# --- 2. CONFIGURACIÓN DE IA (CORREGIDA) ---
instrucciones_ia = "Eres un experto en econometría aplicada. Ayuda al usuario a interpretar resultados estadísticos."

if "messages" not in st.session_state:
    st.session_state.messages = []

# Intentar inicializar el modelo con el nombre correcto
model_ia = None
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Cambiamos a 'gemini-1.5-flash' asegurando el formato
        model_ia = genai.GenerativeModel(
            model_name='models/gemini-1.5-flash', 
            system_instruction=instrucciones_ia
        )
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo de IA: {e}")
else:
    st.sidebar.warning("Configura tu GEMINI_API_KEY para usar el asistente.")

# --- 3. FILTROS Y DATOS ---
st.sidebar.header("Parámetros")
ticker = st.sidebar.text_input("Ticker", "AAPL")
frecuencia_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
mapa_resample = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "YE"}

opcion = st.sidebar.radio("Sección:", ["Análisis", "Econometría", "Balances"])

@st.cache_data
def obtener_datos(ticker, freq):
    data = yf.download([ticker, "SPY"], period="2y")['Close']
    return data.resample(mapa_resample[freq]).last().dropna()

try:
    df = obtener_datos(ticker, frecuencia_label)
    asset_series = df[ticker]
    ret_asset = np.log(asset_series / asset_series.shift(1)).dropna()
    ret_spy = np.log(df["SPY"] / df["SPY"].shift(1)).dropna()

    # --- 4. INTERFAZ ---
    if opcion == "Análisis":
        st.plotly_chart(px.line(asset_series, title=f"Evolución {ticker}"), use_container_width=True)
        st.plotly_chart(px.histogram(ret_asset, title="Distribución de Retornos", marginal="box"), use_container_width=True)

    elif opcion == "Econometría":
        # Aseguramos que los retornos tengan la misma longitud para OLS
        df_ret = pd.concat([ret_asset, ret_spy], axis=1).dropna()
        X = sm.add_constant(df_ret["SPY"])
        res_ols = sm.OLS(df_ret[ticker], X).fit()
        st.write(f"**Beta:** {res_ols.params[1]:.4f}")
        st.text(res_ols.summary())

    # --- 5. CHATBOT FLOTANTE (POPOVER) ---
    with st.sidebar:
        st.markdown("---")
        with st.popover("💬 Consultar Asistente"):
            st.write("### Asistente Econométrico")
            
            chat_container = st.container(height=350)
            for m in st.session_state.messages:
                chat_container.chat_message(m["role"]).write(m["content"])

            if prompt := st.chat_input("¿Qué significa mi Beta?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                chat_container.chat_message("user").write(prompt)
                
                if model_ia:
                    try:
                        # Inyectamos contexto real del análisis al chat
                        # Solo calculamos el beta si hay datos
                        beta_val = sm.OLS(ret_asset, sm.add_constant(ret_spy)).fit().params[1]
                        contexto = f"Activo: {ticker}. Beta actual: {beta_val:.4f}. Pregunta: {prompt}"
                        
                        response = model_ia.generate_content(contexto)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        chat_container.chat_message("assistant").write(response.text)
                    except Exception as e:
                        st.error(f"Error en la IA: {e}")
                else:
                    st.error("IA no disponible.")

except Exception as e:
    st.error(f"Error: {e}")