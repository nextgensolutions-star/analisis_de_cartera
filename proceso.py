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

st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

# 2. INICIALIZACIÓN DE IA
def setup_ai():
    if "GEMINI_API_KEY" not in st.secrets:
        return None, "Clave no configurada en Secrects.",[]
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
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
            return model, f"✅ IA Activa: {target_model.split('/')[-1]}", available_models
        return None, f"❌ Modelos disponibles: {len(available_models)}, pero ninguno compatible.", available_models
    except Exception as e:
        return None, f"⚠️ Error IA: {str(e)}", []

if "messages" not in st.session_state:
    st.session_state.messages = []

model_ia, status_ia, models_list = setup_ai()

# 3. BARRA LATERAL: CONFIGURACIÓN
st.sidebar.header("📊 Parámetros de Análisis")
st.sidebar.info(status_ia)

# Botón de modo oscuro
modo_oscuro = st.sidebar.toggle("🌙 Modo Oscuro", value=False)

if modo_oscuro:
    dark_mode_css = """
    <style>
    .stApp { background-color: #0e1117 !important; color: #fafafa !important; }
    header[data-testid="stHeader"] { background-color: #0e1117 !important; }
    [data-testid="stToolbar"] { background-color: #0e1117 !important; }
    [data-testid="stDecoration"] { background-color: #0e1117 !important; }
    [data-testid="stSidebar"] { background-color: #1a1d29 !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #fafafa !important; }
    [data-testid="stSidebar"] button { background-color: #ff6b35 !important; color: #0e1117 !important; border: none !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] button:hover { background-color: #ff8c42 !important; }
    [data-testid="stMetric"] { background-color: #1a1d29 !important; border: 1px solid #2d3142 !important; }
    [data-testid="stMetricLabel"] { color: #ff4444 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #ff6b35 !important; font-weight: bold !important; }
    h1 { color: #00d4ff !important; font-weight: bold !important; }
    h2 { color: #00d4ff !important; font-weight: bold !important; }
    h3 { color: #00d4ff !important; }
    p { color: #e0e0e0 !important; }
    </style>
    """
    st.markdown(dark_mode_css, unsafe_allow_html=True)

# Base de datos de tickers
TICKERS_DB = {
    "Acciones": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc. (Google)",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "BRK-B": "Berkshire Hathaway",
        "V": "Visa Inc.",
        "JPM": "JPMorgan Chase",
        "WMT": "Walmart Inc.",
        "MA": "Mastercard Inc.",
        "PG": "Procter & Gamble",
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group",
        "HD": "Home Depot",
        "BAC": "Bank of America",
        "XOM": "Exxon Mobil",
        "ORCL": "Oracle Corporation",
        "KO": "Coca-Cola Company",
        "PEP": "PepsiCo Inc.",
        "COST": "Costco Wholesale",
        "NFLX": "Netflix Inc.",
        "DIS": "Walt Disney Company",
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices",
        "NKE": "Nike Inc.",
        "PYPL": "PayPal Holdings",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc."
    },
    "Criptos": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "BNB-USD": "Binance Coin",
        "XRP-USD": "Ripple",
        "ADA-USD": "Cardano",
        "DOGE-USD": "Dogecoin",
        "SOL-USD": "Solana",
        "DOT-USD": "Polkadot",
        "MATIC-USD": "Polygon",
        "AVAX-USD": "Avalanche",
        "LINK-USD": "Chainlink",
        "UNI-USD": "Uniswap"
    },
    "Forex": {
        "EURUSD=X": "Euro / US Dollar",
        "GBPUSD=X": "British Pound / US Dollar",
        "USDJPY=X": "US Dollar / Japanese Yen",
        "AUDUSD=X": "Australian Dollar / US Dollar",
        "USDCAD=X": "US Dollar / Canadian Dollar",
        "USDCHF=X": "US Dollar / Swiss Franc",
        "NZDUSD=X": "New Zealand Dollar / US Dollar",
        "EURGBP=X": "Euro / British Pound",
        "EURJPY=X": "Euro / Japanese Yen"
    },
    "Índices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index",
        "^FTSE": "FTSE 100 (UK)",
        "^GDAXI": "DAX (Germany)",
        "^N225": "Nikkei 225 (Japan)",
        "^HSI": "Hang Seng (Hong Kong)"
    }
}

tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acciones", "Criptos", "Forex", "Índices"])

ticker_options = [f"{ticker} - {nombre}" for ticker, nombre in TICKERS_DB[tipo_activo].items()]

ticker_seleccionado = st.sidebar.selectbox(
    "Buscar Ticker",
    options=ticker_options,
    index=0,
    help="Empieza a escribir para buscar"
)

ticker_user = ticker_seleccionado.split(" - ")[0]

usar_personalizado = st.sidebar.checkbox("✏️ Escribir ticker manualmente")
if usar_personalizado:
    ticker_user = st.sidebar.text_input("Ticker personalizado", value=ticker_user).upper()

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    f_inicio = st.date_input("Desde", datetime.now() - timedelta(days=365*2))
with col_f2:
    f_fin = st.date_input("Hasta", datetime.now())

f_label = st.sidebar.selectbox("Frecuencia", ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"])
m_resample = {"Diario": "D", "Semanal": "W", "Mensual": "ME", "Trimestral": "QE", "Anual": "YE"}

seccion = st.sidebar.radio("Navegación", ["📈 Análisis de Mercado", "📊 Econometría", "📑 Balances Contables", "💬 Asistente IA"])

# 4. CARGA DE DATOS
@st.cache_data
def get_data(ticker, start, end, freq):
    try:
        df = yf.download([ticker, "SPY"], start=start, end=end, progress=False)
        
        if 'Close' in df.columns:
            df_close = df['Close']
        else:
            df_close = df
        
        if ticker not in df_close.columns or "SPY" not in df_close.columns:
            raise ValueError(f"No se pudieron descargar datos para {ticker} o SPY")
        
        df_resampled = df_close.resample(m_resample[freq]).last().dropna()
        return df_resampled
    except Exception as e:
        raise Exception(f"Error descargando datos: {str(e)}")

# 5. LÓGICA PRINCIPAL
try:
    data = get_data(ticker_user, f_inicio, f_fin, f_label)
    
    if ticker_user not in data.columns:
        st.error(f"❌ No se encontraron datos para {ticker_user}. Verifica que el ticker sea correcto.")
        st.stop()
    
    if "SPY" not in data.columns:
        st.error("❌ No se pudieron descargar datos de SPY para comparación.")
        st.stop()
    
    asset_p = data[ticker_user]
    spy_p = data["SPY"]
    
    ret_asset = np.log(asset_p / asset_p.shift(1)).dropna()
    ret_spy = np.log(spy_p / spy_p.shift(1)).dropna()
    df_ret = pd.concat([ret_asset, ret_spy], axis=1).dropna()

    st.title(f"Plataforma de Análisis: {ticker_user}")

    if seccion == "📈 Análisis de Mercado":
        X_beta = sm.add_constant(ret_spy)
        modelo_beta = sm.OLS(ret_asset, X_beta).fit()
        beta_valor = modelo_beta.params[1]
        
        rend_total = (asset_p.iloc[-1] / asset_p.iloc[0]) - 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimiento Período", f"{rend_total:.2%}")
        c2.metric("Precio Cierre", f"{asset_p.iloc[-1]:.2f}")
        c3.metric("Volatilidad (Std)", f"{ret_asset.std():.4f}")
        c4.metric("Beta vs S&P 500", f"{beta_valor:.4f}", 
                 help="β > 1: Más volátil que el mercado | β < 1: Menos volátil | β < 0: Correlación inversa")

        st.markdown("## 📊 Visualización de Datos")
        
        if modo_oscuro:
            fig1 = px.line(asset_p, title="Evolución del Precio")
            fig1.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font_color='#ff4444',
                title_font_color='#ff4444',
                xaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                yaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                height=400
            )
            fig1.update_traces(line_color='#ff6b35')
        else:
            fig1 = px.line(asset_p, title="Evolución del Precio")
        st.plotly_chart(fig1, use_container_width=True)
        
        if modo_oscuro:
            fig2 = px.line(ret_asset, title="Retornos Logarítmicos")
            fig2.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font_color='#ff4444',
                title_font_color='#ff4444',
                xaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                yaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                height=400
            )
            fig2.update_traces(line_color='#ff6b35')
        else:
            fig2 = px.line(ret_asset, title="Retornos Logarítmicos")
        st.plotly_chart(fig2, use_container_width=True)
        
        if modo_oscuro:
            fig3 = px.histogram(ret_asset, title="Distribución de Retornos", marginal="box")
            fig3.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font_color='#ff4444',
                title_font_color='#ff4444',
                xaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                yaxis=dict(gridcolor='#2d3142', color='#ff4444'),
                height=400
            )
            fig3.update_traces(marker_color='#ff6b35')
        else:
            fig3 = px.histogram(ret_asset, title="Distribución de Retornos", marginal="box")
        st.plotly_chart(fig3, use_container_width=True)

    elif seccion == "📊 Econometría":
        st.subheader("Modelos de Regresión y Estacionariedad")
        
        X = sm.add_constant(df_ret["SPY"])
        res_ols = sm.OLS(df_ret[ticker_user], X).fit()
        
        col_ols, col_adf = st.columns([2, 1])
        
        with col_ols:
            st.markdown("**Regresión vs S&P 500 (Cálculo del Beta)**")
            st.text(res_ols.summary())
        
        with col_adf:
            st.markdown("**Test de Dickey-Fuller**")
            p_val = adfuller(asset_p)[1]
            st.metric("p-value ADF", f"{p_val:.4f}")
            if p_val < 0.05:
                st.success("✅ Serie Estacionaria")
            else:
                st.warning("⚠️ Serie No Estacionaria")

    elif seccion == "📑 Balances Contables":
        st.subheader("Información Financiera")
        rep = st.radio("Reporte:", ["Balance Sheet", "Income Statement", "Cash Flow"], horizontal=True)
        t_obj = yf.Ticker(ticker_user)
        df_f = t_obj.balance_sheet if rep == "Balance Sheet" else (t_obj.income_stmt if rep == "Income Statement" else t_obj.cashflow)
        st.dataframe(df_f, use_container_width=True)

    elif seccion == "💬 Asistente IA":
        st.subheader("🤖 Profesor de Econometría Virtual")
        st.write("Pregúntame sobre interpretación de resultados, conceptos estadísticos o análisis econométricos.")
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        
        if prompt := st.chat_input("Escribe tu pregunta aquí..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if model_ia:
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            beta_val = sm.OLS(ret_asset, sm.add_constant(ret_spy)).fit().params[1]
                            context = f"Contexto: Ticker {ticker_user}, Beta {beta_val:.4f}. Pregunta: {prompt}"
                            
                            resp = model_ia.generate_content(context)
                            respuesta_texto = resp.text
                            
                            st.write(respuesta_texto)
                            st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
                            
                        except Exception as e:
                            error_msg = f"Error al generar respuesta: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
            else:
                with st.chat_message("assistant"):
                    msg = "⚠️ IA no disponible. Verifica que tu API key esté configurada correctamente en Streamlit Secrets."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

    with st.sidebar:
        st.markdown("---")
        st.subheader("📥 Exportar Datos")
        st.download_button("Descargar CSV", data=data.to_csv().encode('utf-8'), file_name=f"{ticker_user}.csv")
        
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=True)
        st.download_button("Descargar Excel", data=out.getvalue(), file_name=f"{ticker_user}.xlsx")

except Exception as e:
    st.error(f"Error en la terminal: {e}")
