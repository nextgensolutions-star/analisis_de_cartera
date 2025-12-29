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

# CSS TOTALMENTE PERSONALIZADO
st.markdown("""
    <style>
    /* 1. Fondo general oscuro y barra superior */
    .stApp, header[data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    /* 2. Títulos específicos en Celeste (#00d4ff) */
    /* Título Principal (h1), Subtítulos (h2) y Título de Sidebar */
    h1, h2, [data-testid="stSidebar"] h2 {
        color: #00d4ff !format !important;
    }
    
    /* Forzar celeste en el texto de 'Visualización de Datos' */
    .celeste-title {
        color: #00d4ff !important;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* 3. Estilo de la Barra Lateral */
    [data-testid="stSidebar"] {
        background-color: #1a1d29 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    /* 4. Métricas en modo oscuro */
    [data-testid="stMetric"] {
        background-color: #1a1d29 !important;
        border: 1px solid #2d3142 !important;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] { color: #00d4ff !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INICIALIZACIÓN DE IA ---
def setup_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None, "⚠️ Clave no configurada.", []
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = next((m for m in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro'] if m in available_models), available_models[0] if available_models else "")
        if target_model:
            model = genai.GenerativeModel(model_name=target_model)
            return model, f"✅ IA Activa: {target_model.split('/')[-1]}", available_models
    except:
        pass
    return None, "⚠️ Error IA", []

model_ia, status_ia, _ = setup_ai()

# --- 3. BARRA LATERAL ---
st.sidebar.markdown("## Parámetros de Análisis")
st.sidebar.info(status_ia)

TICKERS_DB = {
    "Acciones": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "GOOGL": "Alphabet"},
    "Criptos": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
    "Forex": {"EURUSD=X": "Euro/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/Yen"},
    "Índices": {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^VIX": "VIX"}
}

tipo_activo = st.sidebar.selectbox("Tipo de Activo", list(TICKERS_DB.keys()))
ticker_opts = [f"{t} - {n}" for t, n in TICKERS_DB[tipo_activo].items()]
ticker_sel = st.sidebar.selectbox("Buscar Ticker", options=ticker_opts)
ticker_user = ticker_sel.split(" - ")[0]

if st.sidebar.checkbox("✏️ Ticker manual"):
    ticker_user = st.sidebar.text_input("Escribir Ticker", value=ticker_user).upper()

f_inicio = st.sidebar.date_input("Desde", datetime.now() - timedelta(days=365*2))
f_fin = st.sidebar.date_input("Hasta", datetime.now())
f_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual"])
m_resample = {"Diario": "D", "Semanal": "W", "Mensual": "ME"}

seccion = st.sidebar.radio("Navegación", ["📈 Análisis de Mercado", "📊 Econometría", "💬 Asistente IA"])

# --- 4. CARGA DE DATOS ---
@st.cache_data
def get_data(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end, progress=False)
    df_close = df['Close'] if 'Close' in df.columns else df
    return df_close.resample(m_resample[freq]).last().dropna()

# --- 5. LÓGICA PRINCIPAL ---
try:
    data = get_data(ticker_user, f_inicio, f_fin, f_label)
    asset_p = data[ticker_user]
    spy_p = data["SPY"]
    ret_asset = np
