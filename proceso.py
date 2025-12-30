import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import io

# 1. CONFIGURACIÓN GENERAL
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")

st.markdown("""
<style>
.stMetric {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# 2. BARRA LATERAL
st.sidebar.header("📊 Parámetros de Análisis")

modo_oscuro = st.sidebar.toggle("🌙 Modo Oscuro", value=False)

if modo_oscuro:
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    header, [data-testid="stSidebar"] { background-color: #1a1d29; }
    </style>
    """, unsafe_allow_html=True)

# 3. BASE DE TICKERS
TICKERS_DB = {
    "Acciones": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla"
    },
    "Criptos": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum"
    },
    "Forex": {
        "EURUSD=X": "EUR/USD",
        "USDJPY=X": "USD/JPY"
    },
    "Índices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones"
    }
}

tipo_activo = st.sidebar.selectbox("Tipo de Activo", list(TICKERS_DB.keys()))

ticker_opciones = [
    f"{k} - {v}" for k, v in TICKERS_DB[tipo_activo].items()
]

ticker_sel = st.sidebar.selectbox("Ticker", ticker_opciones)
ticker = ticker_sel.split(" - ")[0]

f_inicio = st.sidebar.date_input("Desde", datetime.now() - timedelta(days=365 * 2))
f_fin = st.sidebar.date_input("Hasta", datetime.now())

frecuencia = st.sidebar.selectbox("Frecuencia", ["Diario", "Mensual"])
map_freq = {"Diario": "D", "Mensual": "M"}

seccion = st.sidebar.radio(
    "Navegación",
    ["📈 Análisis de Mercado", "📊 Econometría", "📑 Balances Contables"]
)

# 4. CARGA DE DATOS
@st.cache_data
def get_data(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end, progress=False)
    df = df["Close"].resample(map_freq[freq]).last().dropna()
    return df

try:
    data = get_data(ticker, f_inicio, f_fin, frecuencia)

    asset = data[ticker]
    spy = data["SPY"]

    ret_asset = np.log(asset / asset.shift(1)).dropna()
    ret_spy = np.log(spy / spy.shift(1)).dropna()

    df_ret = pd.concat([ret_asset, ret_spy], axis=1).dropna()
    df_ret.columns = ["asset", "market"]

    st.title(f"Terminal de Análisis: {ticker}")

    # 5. SECCIONES
    if seccion == "📈 Análisis de Mercado":
        X = sm.add_constant(df_ret["market"])
        modelo = sm.OLS(df_ret["asset"], X).fit()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimiento", f"{asset.iloc[-1]/asset.iloc[0]-1:.2%}")
        c2.metric("Precio Final", f"{asset.iloc[-1]:.2f}")
        c3.metric("Volatilidad", f"{ret_asset.std():.4f}")
        c4.metric("Beta", f"{modelo.params[1]:.4f}")

        st.plotly_chart(px.line(asset, title="Precio"), use_container_width=True)
        st.plotly_chart(px.line(ret_asset, title="Retornos"), use_container_width=True)
        st.plotly_chart(px.histogram(ret_asset, title="Distribución"), use_container_width=True)

    elif seccion == "📊 Econometría":
        st.subheader("Regresión vs Mercado")
        st.text(sm.OLS(df_ret["asset"], sm.add_constant(df_ret["market"])).fit().summary())

        st.subheader("Test Dickey-Fuller")
        p_val = adfuller(asset)[1]
        st.metric("p-value", f"{p_val:.4f}")
        if p_val < 0.05:
            st.success("Serie estacionaria")
        else:
            st.warning("Serie no estacionaria")

    elif seccion == "📑 Balances Contables":
        t = yf.Ticker(ticker)
        rep = st.radio("Reporte", ["Balance", "Resultados", "Cash Flow"], horizontal=True)

        if rep == "Balance":
            st.dataframe(t.balance_sheet)
        elif rep == "Resultados":
            st.dataframe(t.income_stmt)
        else:
            st.dataframe(t.cashflow)

    # 6. EXPORTACIÓN
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Exportar")
    st.sidebar.download_button(
        "CSV",
        data=data.to_csv().encode(),
        file_name=f"{ticker}.csv"
    )

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        data.to_excel(writer)
    st.sidebar.download_button(
        "Excel",
        data=out.getvalue(),
        file_name=f"{ticker}.xlsx"
    )

except Exception as e:
    st.error(f"Error: {e}")
