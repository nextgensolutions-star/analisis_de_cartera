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

# CSS para el botón flotante y diseño de tarjetas
st.markdown("""
    <style>
    .stPopover { width: 100%; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    [data-testid="stSidebarNav"] { padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INICIALIZACIÓN DE IA (Detección automática de modelo) ---
def setup_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None, "⚠️ Clave no configurada en Secrets."
    
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Buscamos qué modelos tenés habilitados para evitar el error 404
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Prioridad de modelos disponibles en diciembre 2025
        target_model = ""
        for m_name in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
            if m_name in available_models:
                target_model = m_name
                break
        
        if not target_model and available_models:
            target_model = available_models[0]
            
        if target_model:
            model = genai.GenerativeModel(
                model_name=target_model,
                system_instruction="Eres un experto en econometría aplicada. Explica resultados de OLS y ADF de forma académica."
            )
            return model, f"✅ IA Activa: {target_model.split('/')[-1]}"
        return None, "❌ No se encontraron modelos disponibles."
    except Exception as e:
        return None, f"⚠️ Error IA: {str(e)}"

if "messages" not in st.session_state:
    st.session_state.messages = []

model_ia, status_ia = setup_ai()

# --- 3. BARRA LATERAL: CONFIGURACIÓN ---
st.sidebar.header("📊 Parámetros de Análisis")
st.sidebar.info(status_ia)

tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acciones", "Criptos", "Forex", "Índices"])
ticker_map = {"Acciones": "AAPL", "Criptos": "BTC-USD", "Forex": "EURUSD=X", "Índices": "^GSPC"}
ticker_user = st.sidebar.text_input("Ticker", ticker_map[tipo_activo])

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    f_inicio = st.date_input("Desde", datetime.now() - timedelta(days=365*2))
with col_f2:
    f_fin = st.date_input("Hasta", datetime.now())

f_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
m_resample = {"Diario": "D", "Semanal": "W", "Mensual": "ME", "Trimestral": "QE", "Anual": "YE"}

seccion = st.sidebar.radio("Navegación", ["📈 Análisis de Mercado", "📊 Econometría", "📑 Balances Contables"])

# --- 4. CARGA DE DATOS ---
@st.cache_data
def get_data(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end)['Close']
    return df.resample(m_resample[freq]).last().dropna()

# --- 5. LÓGICA DE CÁLCULO Y VISUALIZACIÓN ---
try:
    data = get_data(ticker_user, f_inicio, f_fin, f_label)
    asset_p = data[ticker_user]
    spy_p = data["SPY"]
    
    # Retornos para análisis
    ret_asset = np.log(asset_p / asset_p.shift(1)).dropna()
    ret_spy = np.log(spy_p / spy_p.shift(1)).dropna()
    df_ret = pd.concat([ret_asset, ret_spy], axis=1).dropna()

    st.title(f"Plataforma de Análisis: {ticker_user}")

    if seccion == "📈 Análisis de Mercado":
        # Métricas
        rend_total = (asset_p.iloc[-1] / asset_p.iloc[0]) - 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendimiento Período", f"{rend_total:.2%}")
        c2.metric("Precio Cierre", f"{asset_p.iloc[-1]:.2f}")
        c3.metric("Volatilidad (Std)", f"{ret_asset.std():.4f}")

        # Gráficos
        tab1, tab2 = st.tabs(["Precios y Retornos", "Distribución Estadística"])
        with tab1:
            st.plotly_chart(px.line(asset_p, title="Evolución del Precio"), use_container_width=True)
            st.plotly_chart(px.line(ret_asset, title="Retornos Logarítmicos"), use_container_width=True)
        with tab2:
            st.plotly_chart(px.histogram(ret_asset, title="Histograma y Outliers", marginal="box"), use_container_width=True)

    elif seccion == "📊 Econometría":
        st.subheader("Modelos de Regresión y Estacionariedad")
        col_ols, col_adf = st.columns([2, 1])
        
        with col_ols:
            st.markdown("**Regresión vs S&P 500 (Cálculo del Beta)**")
            X = sm.add_constant(df_ret["SPY"])
            res_ols = sm.OLS(df_ret[ticker_user], X).fit()
            st.text(res_ols.summary())
        
        with col_adf:
            st.markdown("**Test de Dickey-Fuller**")
            p_val = adfuller(asset_p)[1]
            st.metric("p-value ADF", f"{p_val:.4f}")
            if p_val < 0.05:
                st.success("Serie Estacionaria")
            else:
                st.warning("Serie No Estacionaria")

    elif seccion == "📑 Balances Contables":
        st.subheader("Información Financiera")
        rep = st.radio("Reporte:", ["Balance Sheet", "Income Statement", "Cash Flow"], horizontal=True)
        t_obj = yf.Ticker(ticker_user)
        df_f = t_obj.balance_sheet if rep == "Balance Sheet" else (t_obj.income_stmt if rep == "Income Statement" else t_obj.cashflow)
        st.dataframe(df_f, use_container_width=True)

    # --- 6. CHATBOT FLOTANTE Y EXPORTACIÓN ---
    with st.sidebar:
        st.markdown("---")
        with st.popover("💬 Preguntar al Asistente IA"):
            st.write("#### Profesor de Econometría Virtual")
            chat_box = st.container(height=300)
            
            for m in st.session_state.messages:
                chat_box.chat_message(m["role"]).write(m["content"])

            if p := st.chat_input("¿Cómo interpreto el p-value?"):
                st.session_state.messages.append({"role": "user", "content": p})
                chat_box.chat_message("user").write(p)
                
                if model_ia:
                    # Contexto para evitar respuestas genéricas
                    beta_val = sm.OLS(ret_asset, sm.add_constant(ret_spy)).fit().params[1]
                    context = f"Contexto: Ticker {ticker_user}, Beta {beta_val:.4f}. Pregunta: {p}"
                    resp = model_ia.generate_content(context)
                    st.session_state.messages.append({"role": "assistant", "content": resp.text})
                    st.rerun()
                else:
                    st.error("IA no disponible.")

        # Exportar
        st.subheader("📥 Exportar Datos")
        st.download_button("Descargar CSV", data=data.to_csv().encode('utf-8'), file_name=f"{ticker_user}.csv")
        
        # Función para Excel
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=True)
        st.download_button("Descargar Excel", data=out.getvalue(), file_name=f"{ticker_user}.xlsx")

except Exception as e:
    st.error(f"Error en la terminal: {e}")