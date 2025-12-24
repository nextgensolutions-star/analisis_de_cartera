import streamlit as st
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import numpy as np

st.title("Plataforma de Análisis Econométrico")

# 1. Configuración de parámetros
ticker = st.text_input("Ingresa el Ticker (ej: AAPL, TSLA, GGAL)", "AAPL")
metodo = st.selectbox("Elige el método", ["Precios Históricos", "Retornos Log", "Regresión OLS (vs SPY)"])

# 2. Descarga de datos (activo + benchmark para OLS)
data = yf.download([ticker, "SPY"], period="1y")['Close']

if metodo == "Precios Históricos":
    fig = px.line(data, y=ticker, title=f"Precio de Cierre de {ticker}")
    st.plotly_chart(fig)

elif metodo == "Retornos Log":
    # Cálculo de Retornos Logarítmicos: ln(Pt / Pt-1)
    # Usamos LaTeX para la fórmula
    st.latex(r"r_t = \ln(P_t) - \ln(P_{t-1})")
    data['Log_Ret'] = np.log(data[ticker] / data[ticker].shift(1))
    fig = px.histogram(data, x='Log_Ret', title=f"Distribución de Retornos de {ticker}")
    st.plotly_chart(fig)

elif metodo == "Regresión OLS (vs SPY)":
    st.subheader("Modelo de Valuación de Activos (CAPM - Beta)")
    
    # Preparamos los retornos para la regresión
    retornos = np.log(data / data.shift(1)).dropna()
    
    Y = retornos[ticker]     # Variable dependiente
    X = retornos["SPY"]      # Variable independiente
    X = sm.add_constant(X)   # Añadimos la constante (intercepto)
    
    # Ajustamos el modelo
    modelo = sm.OLS(Y, X).fit()
    
    # Mostramos resultados
    st.write(f"**Beta del activo:** {round(modelo.params[1], 4)}")
    st.write(f"**R-Cuadrado:** {round(modelo.rsquared, 4)}")
    
    # Tabla resumen detallada
    st.text("Resumen estadístico del modelo:")
    st.write(modelo.summary())