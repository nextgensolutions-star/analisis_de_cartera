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

# 1. CONFIGURACIÓN Y ESTILOS
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")

# CSS mejorado para cumplir con tus requisitos de color y barra superior
css_custom = """
    <style>
    /* Estilo general de métricas */
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    
    /* Forzar que la barra superior sea oscura siempre o según el modo */
    header[data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0e1117 !important;
        color: white !important;
    }
    
    /* Títulos en celeste por defecto (h1 y h2) */
    h1, h2, [data-testid="stSidebar"] h2 {
        color: #00d4ff !important;
        font-weight: bold !important;
    }
    </style>
    """
st.markdown(css_custom, unsafe_allow_html=True)

# --- 2. INICIALIZACIÓN DE IA ---
def setup_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None, "⚠️ Clave no configurada.", []
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = next((m for m in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if m in available_models), available_models[0] if available_models else "")
        if target_model:
            model = genai.GenerativeModel(model_name=target_model, system_instruction="Experto en econometría.")
            return model, f"✅ IA Activa: {target_model.split('/')[-1]}", available_models
        return None, "❌ No hay modelos disponibles.", available_models
    except Exception as e:
        return None, f"⚠️ Error IA: {str(e)}", []

if "messages" not in st.session_state:
    st.session_state.messages = []
model_ia, status_ia, models_list = setup_ai()

# --- 3. BARRA LATERAL ---
# Cambié st.sidebar.header por un markdown con ID específico si fuera necesario, 
# pero el CSS de arriba ya captura los h2 de la barra lateral.
st.sidebar.markdown("## 📊 Parámetros de Análisis")
st.sidebar.info(status_ia)

modo_oscuro = st.sidebar.toggle("🌙 Modo Oscuro", value=True)

if modo_oscuro:
    dark_mode_css = """
    <style>
    .stApp { background-color: #0e1117 !important; color: #fafafa !important; }
    [data-testid="stSidebar"] { background-color: #1a1d29 !important; }
    [data-testid="stMetric"] { background-color: #1a1d29 !important; border: 1px solid #2d3142 !important; }
    /* Títulos Celeste en Modo Oscuro */
    h1, h2, h3, [data-testid="stSidebar"] h2, [data-testid="stWidgetLabel"] p { color: #00d4ff !important; }
    </style>
    """
    st.markdown(dark_mode_css, unsafe_allow_html=True)

# Base de datos de tickers (Simplificada para brevedad)
TICKERS_DB = {
    "Acciones": {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA"},
    "Criptos": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
    "Forex": {"EURUSD=X": "Euro/USD"},
    "Índices": {"^GSPC": "S&P 500", "^IXIC": "NASDAQ"}
}

tipo_activo = st.sidebar.selectbox("Tipo de Activo", list(TICKERS_DB.keys()))
ticker_options = [f"{t} - {n}" for t, n in TICKERS_DB[tipo_activo].items()]
ticker_seleccionado = st.sidebar.selectbox("Buscar Ticker", options=ticker_options)
ticker_user = ticker_seleccionado.split(" - ")[0]

if st.sidebar.checkbox("✏️ Escribir ticker manualmente"):
    ticker_user = st.sidebar.text_input("Ticker personalizado", value=ticker_user).upper()

col_f1, col_f2 = st.sidebar.columns(2)
f_inicio = col_f1.date_input("Desde", datetime.now() - timedelta(days=365*2))
f_fin = col_f2.date_input("Hasta", datetime.now())
f_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual"])
m_resample = {"Diario": "D", "Semanal": "W", "Mensual": "ME"}

seccion = st.sidebar.radio("Navegación", ["📈 Análisis de Mercado", "📊 Econometría", "📑 Balances", "💬 Asistente IA"])

# --- 4. CARGA Y LÓGICA ---
@st.cache_data
def get_data(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end, progress=False)
    df_close = df['Close'] if 'Close' in df.columns else df
    return df_close.resample(m_resample[freq]).last().dropna()

try:
    data = get_data(ticker_user, f_inicio, f_fin, f_label)
    asset_p = data[ticker_user]
    spy_p = data["SPY"]
    ret_asset = np.log(asset_p / asset_p.shift(1)).dropna()
    ret_spy = np
