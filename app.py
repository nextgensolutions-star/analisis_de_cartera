import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# 1. Configuración de la página
st.set_page_config(page_title="Mi Cartera - Bolsa Argentina", layout="wide", page_icon="₱")

# --- ESTILOS CSS (Tu diseño original) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    :root {{
      --bg-primary: #0a0e27;
      --bg-secondary: #151b3d;
      --accent-green: #00ff88;
      --accent-red: #ff4757;
      --accent-blue: #4da6ff;
      --text-primary: #ffffff;
      --text-muted: #718096;
      --border: #2d3748;
    }}

    /* Escondemos elementos nativos de Streamlit para que luzca como tu HTML */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{padding: 2rem; background-color: var(--bg-primary);}}

    .stat-card {{
      background: var(--bg-secondary);
      padding: 25px;
      border-radius: 16px;
      border: 1px solid var(--border);
      position: relative;
      margin-bottom: 20px;
    }}
    .stat-card::before {{
      content: '';
      position: absolute; top: 0; left: 0; width: 100%; height: 4px;
      background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
    }}
    .stat-value {{
      font-size: 32px; font-weight: 700;
      color: var(--accent-green);
      font-family: 'Space Grotesk', sans-serif;
    }}
    .stat-label {{
      font-size: 12px; text-transform: uppercase; letter-spacing: 2px;
      color: var(--text-muted); font-family: 'IBM Plex Mono', monospace;
    }}
</style>
""", unsafe_allow_html=True)

# --- LÓGICA DE DATOS ---
def load_data():
    # Cargar credenciales desde secrets (mismo formato que GitHub Actions)
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
    
    # Autenticar con Google
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    # Abrir el sheet
    SHEET_ID = st.secrets["GOOGLE_SHEETS_ID"]
    sheet = client.open_by_key(SHEET_ID)
    
    # Cargar las 3 pestañas
    df_ops = pd.DataFrame(sheet.worksheet("operaciones").get_all_records())
    df_coti = pd.DataFrame(sheet.worksheet("cotizaciones").get_all_records())
    df_maestra = pd.DataFrame(sheet.worksheet("maestra").get_all_records())
    
    # Verificar si las hojas tienen datos
    if df_ops.empty:
        raise ValueError("La hoja 'operaciones' está vacía. Agrega datos.")
    if df_coti.empty:
        raise ValueError("La hoja 'cotizaciones' está vacía. Agrega datos.")
    if df_maestra.empty:
        raise ValueError("La hoja 'maestra' está vacía. Agrega datos.")
    
    # Convertir columnas numéricas (maneja celdas vacías, texto, etc.)
    if 'cantidad' in df_ops.columns:
        df_ops['cantidad'] = pd.to_numeric(df_ops['cantidad'], errors='coerce').fillna(0)
    if 'precio_compra_venta' in df_ops.columns:
        df_ops['precio_compra_venta'] = pd.to_numeric(df_ops['precio_compra_venta'], errors='coerce').fillna(0)
    
    if 'ultimoPrecio' in df_coti.columns:
        df_coti['ultimoPrecio'] = pd.to_numeric(df_coti['ultimoPrecio'], errors='coerce').fillna(0)
    
    if 'beta' in df_maestra.columns:
        df_maestra['beta'] = pd.to_numeric(df_maestra['beta'], errors='coerce').fillna(1)
    
    return df_ops, df_coti, df_maestra

try:
    df_ops, df_coti, df_maestra = load_data()

    # Neteo de Cartera (Cantidades actuales)
    df_neto = df_ops.groupby('simbolo').agg({'cantidad': 'sum', 'precio_compra_venta': 'mean'}).reset_index()
    df_vivos = df_neto[df_neto['cantidad'] > 0].copy()

    # Mezclamos con precios vivos y datos maestros
    df_final = pd.merge(df_vivos, df_coti, on='simbolo', how='left')
    df_final = pd.merge(df_final, df_maestra, on='simbolo', how='left')

    # Cálculos dinámicos
    df_final['valorizado'] = df_final['cantidad'] * df_final['ultimoPrecio']
    patrimonio_total = df_final['valorizado'].sum()
    
    # Beta Ponderado
    if patrimonio_total > 0:
        beta_p = (df_final['beta'] * (df_final['valorizado'] / patrimonio_total)).sum()
    else: beta_p = 0

    # --- RENDERIZADO DEL FRONTEND ---
    
    # Header estilo tu HTML
    st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #2d3748; padding-bottom: 20px; margin-bottom: 40px;'>
            <div>
                <h1 style='color: white; margin: 0;'>Mi Cartera ₱</h1>
                <p style='color: #718096; font-family: "IBM Plex Mono";'>BOLSA ARGENTINA | ANALYST MODE</p>
            </div>
            <div style='background: #151b3d; padding: 10px 20px; border-radius: 50px; border: 1px solid #2d3748;'>
                <span style='color: #00ff88;'>● EN VIVO</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Grid de KPIs (Tus Stat-Cards)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Valor Portfolio</div>
            <div class="stat-value">${patrimonio_total:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Beta Ponderado</div>
            <div class="stat-value">{beta_p:.2f}x</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        # Ganancia latente total
        ganancia = patrimonio_total - (df_final['cantidad'] * df_final['precio_compra_venta']).sum()
        color = "var(--accent-green)" if ganancia >= 0 else "var(--accent-red)"
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Ganancia Latente</div>
            <div class="stat-value" style="color: {color}">${ganancia:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Activos</div>
            <div class="stat-value">{len(df_final)}</div>
        </div>""", unsafe_allow_html=True)

    # Gráficos (Usando la estética de Streamlit que combina con tu azul oscuro)
    col_left, col_right = st.columns(2)
    with col_left:
        st.write("### Distribución por Sector")
        st.pie_chart(df_final.groupby('sector')['valorizado'].sum())
    
    with col_right:
        st.write("### Concentración por Activo")
        st.bar_chart(df_final.set_index('simbolo')['valorizado'])

    # Tabla Detallada
    st.write("### Detalle de Posiciones Abiertas")
    st.dataframe(df_final[['simbolo', 'cantidad', 'ultimoPrecio', 'valorizado', 'sector', 'beta']], use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar datos del Sheets: {e}")
