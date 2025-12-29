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

# CSS para títulos celestes y barra superior oscura
st.markdown("""
    <style>
    /* 1. Títulos en celeste (Principal, Sidebar y Visualización) */
    h1, h2, .stMarkdown h2, [data-testid="stSidebar"] h2 {
        color: #00d4ff !important;
        font-weight: bold !important;
    }
    
    /* 2. Barra superior (Header) en oscuro */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    
    /* Estilo de métricas */
    [data-testid="stMetric"] {
        background-color: #1a1d29 !important;
        border: 1px solid #2d3142 !important;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] { color: #00d4ff !important; }
    
    /* Fondo general oscuro */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INICIALIZACIÓN DE IA ---
def setup_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None, "⚠️ Clave no configurada.", []
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = ""
        for m_name in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']:
            if m_name in available_models:
                target_model = m_name
                break
        if target_model:
            model = genai.GenerativeModel(model_name=target_model)
            return model, f"✅ IA Activa: {target_model.split('/')[-1]}", available_models
        return None, "❌ No hay modelos compatibles.", available_models
    except Exception as e:
        return None, f"⚠️ Error IA: {str(e)}", []

model_ia, status_ia, models_list = setup_ai()

# --- 3. BARRA LATERAL ---
st.sidebar.markdown("## Parámetros de Análisis")
st.sidebar.info(status_ia)

TICKERS_DB = {
    "Acciones": {"AAPL": "Apple Inc.", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla"},
    "Criptos": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
    "Forex": {"EURUSD=X": "Euro/USD", "GBPUSD=X": "GBP/USD"},
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
    if 'Close' in df.columns:
        df_close = df['Close']
    else:
        df_close = df
    return df_close.resample(m_resample[freq]).last().dropna()

# --- 5. LÓGICA PRINCIPAL ---
try:
    data = get_data(ticker_user, f_inicio, f_fin, f_label)
    
    if ticker_user not in data.columns:
        st.error(f"No se encontraron datos para {ticker_user}")
        st.stop()

    asset_p = data[ticker_user]
    spy_p = data["SPY"]
    ret_asset = np.log(asset_p / asset_p.shift(1)).dropna()
    ret_spy = np.log(spy_p / spy_p.shift(1)).dropna()

    st.title(f"Plataforma de Análisis: {ticker_user}")

    if seccion == "📈 Análisis de Mercado":
        # Métricas principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimiento", f"{(asset_p.iloc[-1]/asset_p.iloc[0]-1):.2%}")
        c2.metric("Precio Cierre", f"{asset_p.iloc[-1]:.2f}")
        c3.metric("Volatilidad", f"{ret_asset.std():.4f}")
        
        X_beta = sm.add_constant(ret_spy)
        beta_v = sm.OLS(ret_asset, X_beta).fit().params[1]
        c4.metric("Beta vs S&P 500", f"{beta_v:.4f}")

        st.markdown("## Visualización de Datos")
        
        # GRÁFICOS UNO DEBAJO DEL OTRO
        fig1 = px.line(asset_p, title="Evolución del Precio", template="plotly_dark")
        fig1.update_traces(line_color='#00d4ff')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(ret_asset, title="Retornos Logarítmicos", template="plotly_dark")
        fig2.update_traces(line_color='#ff6b35')
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(ret_asset, title="Distribución de Retornos", marginal="box", template="plotly_dark")
        fig3.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig3, use_container_width=True)

    elif seccion == "📊 Econometría":
        st.markdown("## Análisis Econométrico")
        col_ols, col_adf = st.columns([2, 1])
        
        with col_ols:
            st.markdown("**Regresión OLS vs Mercado**")
            model = sm.OLS(ret_asset, sm.add_constant(ret_spy)).fit()
            st.text(model.summary())
            
        with col_adf:
            st.markdown("**Test Dickey-Fuller (ADF)**")
            p_val = adfuller(asset_p)[1]
            st.metric("p-value ADF", f"{p_val:.4f}")
            if p_val < 0.05:
                st.success("Serie Estacionaria")
            else:
                st.warning("Serie No Estacionaria")

    elif seccion == "💬 Asistente IA":
        st.markdown("## Asistente IA")
        if prompt := st.chat_input("¿Qué deseas analizar?"):
            st.chat_message("user").write(prompt)
            if model_ia:
                with st.spinner("Analizando..."):
                    resp = model_ia.generate_content(f"Sobre {ticker_user}: {prompt}")
                    st.chat_message("assistant").write(resp.text)

except Exception as e:
    st.error(f"Error en la ejecución: {e}")
