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

# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")

# CSS para que el botón del chatbot se vea mejor en la barra lateral
st.markdown("""
    <style>
    .stPopover { width: 100%; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE INTELIGENCIA ARTIFICIAL ---
# Instrucciones para que actúe como profesor de tu Maestría
instrucciones_ia = """
Eres un Asistente Experto de la Maestría en Economía Aplicada. 
Tu misión es interpretar resultados econométricos (Beta, p-values, histogramas) 
con rigor académico pero de forma clara. Cita a autores como Wooldridge o Gujarati 
si el usuario pregunta por la teoría detrás de los modelos.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

model_ia = None
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model_ia = genai.GenerativeModel(
            model_name='gemini-1.5-flash', 
            system_instruction=instrucciones_ia
        )
    except Exception as e:
        st.sidebar.error(f"Error IA: {e}")
else:
    st.sidebar.warning("⚠️ Chatbot: Falta la API Key en los secretos.")

# --- 3. FUNCIONES DE APOYO (EXCEL Y DATOS) ---
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Datos')
    writer.close()
    return output.getvalue()

@st.cache_data
def obtener_datos(ticker, start, end, freq_alias):
    # Descargamos activo y SPY para comparativas econométricas
    df = yf.download([ticker, "SPY"], start=start, end=end)['Close']
    df_res = df.resample(freq_alias).last().dropna()
    return df_res

# --- 4. BARRA LATERAL: FILTROS Y PARÁMETROS ---
st.sidebar.header("⚙️ Configuración")

tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acciones", "Criptos", "Forex", "Índices"])
ticker_map = {"Acciones": "AAPL", "Criptos": "BTC-USD", "Forex": "EURUSD=X", "Índices": "^GSPC"}
ticker_input = st.sidebar.text_input("Ticker", ticker_map[tipo_activo])

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    f_inicio = st.date_input("Desde", datetime.now() - timedelta(days=365*2))
with col_f2:
    f_fin = st.date_input("Hasta", datetime.now())

frecuencia_label = st.sidebar.selectbox("Frecuencia Temporal", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
mapa_resample = {"Diario": "D", "Semanal": "W", "Mensual": "ME", "Trimestral": "QE", "Anual": "YE"}

opcion_menu = st.sidebar.radio("Sección del Dashboard", ["📈 Análisis de Mercado", "📊 Econometría Aplicada", "📑 Datos Fundamentales"])

# --- 5. LÓGICA PRINCIPAL ---
try:
    data = obtener_datos(ticker_input, f_inicio, f_fin, mapa_resample[frecuencia_label])
    asset_series = data[ticker_input]
    spy_series = data["SPY"]
    
    # Cálculos de Retornos Logarítmicos
    ret_asset = np.log(asset_series / asset_series.shift(1)).dropna()
    ret_spy = np.log(spy_series / spy_series.shift(1)).dropna()
    df_retornos = pd.concat([ret_asset, ret_spy], axis=1).dropna()

    st.title(f"Análisis de {ticker_input}")

    if opcion_menu == "📈 Análisis de Mercado":
        # Métricas de cabecera
        rend_total = (asset_series.iloc[-1] / asset_series.iloc[0]) - 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Rendimiento Agregado", f"{rend_total:.2%}")
        c2.metric("Precio Actual", f"{asset_series.iloc[-1]:.2f}")
        c3.metric("Volatilidad (Desv. Est.)", f"{ret_asset.std():.4f}")

        # Pestañas de visualización
        t_lineas, t_dist = st.tabs(["Evolución Temporal", "Distribución Estadística"])
        with t_lineas:
            st.plotly_chart(px.line(asset_series, title=f"Precio de Cierre ({frecuencia_label})"), use_container_width=True)
            st.plotly_chart(px.line(ret_asset, title="Retornos Logarítmicos"), use_container_width=True)
        with t_dist:
            st.write("Análisis de la forma de los retornos (detección de colas pesadas y outliers).")
            st.plotly_chart(px.histogram(ret_asset, title="Histograma de Retornos", marginal="box"), use_container_width=True)

    elif opcion_menu == "📊 Econometría Aplicada":
        st.subheader("Modelado Estadístico")
        col_e1, col_e2 = st.columns([2, 1])
        
        with col_e1:
            st.markdown("**Regresión Lineal OLS (Activo vs S&P 500)**")
            X = sm.add_constant(df_retornos["SPY"])
            modelo = sm.OLS(df_retornos[ticker_input], X).fit()
            st.text(modelo.summary())
        
        with col_e2:
            st.markdown("**Test de Dickey-Fuller (ADF)**")
            res_adf = adfuller(asset_series)
            st.write(f"Estadístico t: `{res_adf[0]:.4f}`")
            st.write(f"p-value: `{res_adf[1]:.4f}`")
            if res_adf[1] < 0.05:
                st.success("La serie es Estacionaria.")
            else:
                st.warning("Serie No Estacionaria (Raíz Unitaria).")

    elif opcion_menu == "📑 Datos Fundamentales":
        st.subheader("Balances y Estados Contables")
        ticker_obj = yf.Ticker(ticker_input)
        tipo_df = st.selectbox("Selecciona el reporte", ["Balance Sheet", "Income Statement", "Cash Flow"])
        
        if tipo_df == "Balance Sheet":
            df_fund = ticker_obj.balance_sheet
        elif tipo_df == "Income Statement":
            df_fund = ticker_obj.income_stmt
        else:
            df_fund = ticker_obj.cashflow
        
        st.dataframe(df_fund, use_container_width=True)

    # --- 6. CHATBOT FLOTANTE Y EXPORTACIÓN (SIDEBAR) ---
    with st.sidebar:
        st.markdown("---")
        # El Chatbot en un Popover para que sea "flotante" sobre el sidebar
        with st.popover("💬 Consultar Asistente IA"):
            st.markdown("### Profesor Virtual de Econometría")
            container_chat = st.container(height=300)
            
            for m in st.session_state.messages:
                container_chat.chat_message(m["role"]).write(m["content"])

            if p_user := st.chat_input("¿Qué significa mi Beta?"):
                st.session_state.messages.append({"role": "user", "content": p_user})
                container_chat.chat_message("user").write(p_user)
                
                if model_ia:
                    # Cálculo rápido del Beta para darle contexto a la IA
                    X_ia = sm.add_constant(df_retornos["SPY"])
                    b_ia = sm.OLS(df_retornos[ticker_input], X_ia).fit().params[1]
                    contexto = f"Activo: {ticker_input}. Beta: {b_ia:.4f}. Pregunta: {p_user}"
                    
                    response = model_ia.generate_content(contexto)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    container_chat.chat_message("assistant").write(response.text)
                else:
                    st.error("IA no disponible.")

        # Botones de Descarga
        st.subheader("📥 Exportar")
        st.download_button("Descargar CSV", data=data.to_csv().encode('utf-8'), file_name=f"{ticker_input}_data.csv")
        st.download_button("Descargar Excel", data=to_excel(data), file_name=f"{ticker_input}_data.xlsx")

except Exception as e:
    st.error(f"Error en la aplicación: {e}")