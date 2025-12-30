import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import io

# ===============================
# CONFIG GENERAL
# ===============================
st.set_page_config(page_title="Terminal Económica Pro", layout="wide")

# ===============================
# PALETA DARK MODE
# ===============================
COLOR_BG = "#0e1117"
COLOR_EJES = "#ffffff"
COLOR_LINEA = "#ff6b35"
COLOR_GRID = "#2d3142"

def aplicar_dark_mode(fig):
    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_EJES),
        xaxis=dict(
            color=COLOR_EJES,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID
        ),
        yaxis=dict(
            color=COLOR_EJES,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID
        )
    )
    return fig

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📊 Parámetros")

modo_oscuro = st.sidebar.toggle("🌙 Modo Noche", value=True)

if modo_oscuro:
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_EJES}; }}
    header, [data-testid="stSidebar"] {{ background-color: #1a1d29; }}
    </style>
    """, unsafe_allow_html=True)

TICKERS_DB = {
    "Acciones": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA"},
    "Criptos": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
    "Forex": {"EURUSD=X": "EUR/USD"},
    "Índices": {"^GSPC": "S&P 500"}
}

tipo = st.sidebar.selectbox("Tipo de activo", list(TICKERS_DB.keys()))
ticker_sel = st.sidebar.selectbox(
    "Ticker",
    [f"{k} - {v}" for k, v in TICKERS_DB[tipo].items()]
)
ticker = ticker_sel.split(" - ")[0]

f_inicio = st.sidebar.date_input("Desde", datetime.now() - timedelta(days=365 * 2))
f_fin = st.sidebar.date_input("Hasta", datetime.now())

freq = st.sidebar.selectbox("Frecuencia", ["Diario", "Mensual"])
MAP_FREQ = {"Diario": "D", "Mensual": "M"}

seccion = st.sidebar.radio(
    "Sección",
    ["📈 Mercado", "📊 Econometría", "📑 Balances"]
)

# ===============================
# DATOS
# ===============================
@st.cache_data
def get_data(ticker, start, end, freq):
    df = yf.download([ticker, "SPY"], start=start, end=end, progress=False)
    df = df["Close"].resample(MAP_FREQ[freq]).last().dropna()
    return df

data = get_data(ticker, f_inicio, f_fin, freq)

if data.empty or ticker not in data.columns or "SPY" not in data.columns:
    st.error("❌ No hay datos suficientes para el rango seleccionado.")
    st.stop()

asset = data[ticker]
spy = data["SPY"]

ret_asset = np.log(asset / asset.shift(1)).dropna()
ret_spy = np.log(spy / spy.shift(1)).dropna()

if ret_asset.empty or ret_spy.empty:
    st.error("❌ No hay datos suficientes para calcular retornos.")
    st.stop()

df_ret = pd.concat([ret_asset, ret_spy], axis=1).dropna()
df_ret.columns = ["asset", "market"]

if df_ret.empty:
    st.error("❌ No hay observaciones comunes activo/mercado.")
    st.stop()

st.title(f"Terminal Económica · {ticker}")

# ===============================
# SECCIONES
# ===============================
if seccion == "📈 Mercado":
    X = sm.add_constant(df_ret["market"])
    modelo = sm.OLS(df_ret["asset"], X).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendimiento", f"{asset.iloc[-1]/asset.iloc[0]-1:.2%}")
    c2.metric("Precio final", f"{asset.iloc[-1]:.2f}")
    c3.metric("Volatilidad", f"{ret_asset.std():.4f}")
    c4.metric("Beta", f"{modelo.params[1]:.4f}")

    fig_p = px.line(asset, title="Precio")
    fig_p.update_traces(line=dict(color=COLOR_LINEA, width=2))
    if modo_oscuro:
        fig_p = aplicar_dark_mode(fig_p)
    st.plotly_chart(fig_p, use_container_width=True)

    fig_r = px.line(ret_asset, title="Retornos")
    fig_r.update_traces(line=dict(color=COLOR_LINEA))
    if modo_oscuro:
        fig_r = aplicar_dark_mode(fig_r)
    st.plotly_chart(fig_r, use_container_width=True)

    fig_h = px.histogram(ret_asset, title="Distribución de retornos")
    fig_h.update_traces(marker_color=COLOR_LINEA)
    if modo_oscuro:
        fig_h = aplicar_dark_mode(fig_h)
    st.plotly_chart(fig_h, use_container_width=True)

elif seccion == "📊 Econometría":
    st.subheader("Regresión vs Mercado")
    st.text(modelo.summary())

    st.subheader("Test Dickey-Fuller")
    p_val = adfuller(asset)[1]
    st.metric("p-value", f"{p_val:.4f}")
    if p_val < 0.05:
        st.success("Serie estacionaria")
    else:
        st.warning("Serie no estacionaria")

elif seccion == "📑 Balances":
    t = yf.Ticker(ticker)
    rep = st.radio("Reporte", ["Balance", "Resultados", "Cash Flow"], horizontal=True)

    if rep == "Balance":
        st.dataframe(t.balance_sheet)
    elif rep == "Resultados":
        st.dataframe(t.income_stmt)
    else:
        st.dataframe(t.cashflow)

# ===============================
# EXPORTAR
# ===============================
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
