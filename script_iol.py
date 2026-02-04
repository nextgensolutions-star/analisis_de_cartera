import os
import json
import requests
import gspread
import pandas as pd
import yfinance as yf
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

# Variables de entorno (GitHub Secrets)
USER = os.environ.get('IOL_USERNAME')
PASS = os.environ.get('IOL_PASSWORD')
SHEET_ID = os.environ.get('GOOGLE_SHEETS_ID')
GOOGLE_CREDS = os.environ.get('GOOGLE_CREDS')

class IOL_Manager:
    def __init__(self):
        self.url_base = "https://api.invertironline.com"
        self.access_token = None

    def login(self):
        url = f"{self.url_base}/token"
        payload = {'username': USER, 'password': PASS, 'grant_type': 'password'}
        response = requests.post(url, data=payload)
        response.raise_for_status()
        self.access_token = response.json().get('access_token')

    def get_operaciones_recientes(self):
        # Buscamos operaciones de los últimos 7 días para asegurar capturar todo
        url = f"{self.url_base}/api/v1/estadocuenta"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        response = requests.get(url, headers=headers)
        # Filtramos solo movimientos que sean compras o ventas de títulos
        movimientos = response.json().get('movimientos', [])
        return movimientos

    def get_precio(self, simbolo):
        """Busca el precio del símbolo en múltiples mercados automáticamente"""
        mercados = ["BCBA", "NYSE", "NASDAQ", "ROFX"]  # Mercados soportados por IOL
        
        for mercado in mercados:
            url = f"{self.url_base}/api/v1/Titulos/{mercado}/Instrumentos/{simbolo}/Cotizacion"
            headers = {'Authorization': f'Bearer {self.access_token}'}
            try:
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data and 'ultimoPrecio' in data:
                        return data
            except:
                continue
        
        return None
    
    def get_precio_yfinance(self, simbolo):
        """Obtiene precio usando yfinance (para criptos y otros activos no disponibles en IOL)"""
        # Mapeo de símbolos comunes a formato yfinance
        simbolo_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'USDT': 'USDT-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
        }
        
        # Si el símbolo está en el mapeo, usar esa versión
        yf_simbolo = simbolo_map.get(simbolo, simbolo)
        
        try:
            ticker = yf.Ticker(yf_simbolo)
            info = ticker.history(period='1d')
            if not info.empty:
                precio = info['Close'].iloc[-1]
                fecha = info.index[-1].strftime('%Y-%m-%dT%H:%M:%S')
                return {
                    'ultimoPrecio': precio,
                    'fechaHora': fecha
                }
        except Exception as e:
            print(f"Error con yfinance para {simbolo}: {e}")
        
        return None
    
    def get_info_fundamental(self, simbolo):
        """Obtiene información fundamental del activo (sector, tipo, beta)"""
        # Mapeo para criptos (beta=1 = neutral, se mueve igual que el mercado)
        crypto_map = {
            'BTC': {'sector': 'Cryptocurrency', 'instrumento': 'Crypto', 'beta': 1},
            'ETH': {'sector': 'Cryptocurrency', 'instrumento': 'Crypto', 'beta': 1},
            'USDT': {'sector': 'Stablecoin', 'instrumento': 'Crypto', 'beta': 0},  # Stablecoin = sin volatilidad
            'ADA': {'sector': 'Cryptocurrency', 'instrumento': 'Crypto', 'beta': 1},
            'SOL': {'sector': 'Cryptocurrency', 'instrumento': 'Crypto', 'beta': 1},
        }
        
        # Si es una crypto conocida, devolver info hardcoded
        if simbolo in crypto_map:
            return crypto_map[simbolo]
        
        # Para acciones, intentar obtener de yfinance
        yf_simbolo = simbolo
        try:
            ticker = yf.Ticker(yf_simbolo)
            info = ticker.info
            
            return {
                'sector': info.get('sector', 'N/A'),
                'instrumento': info.get('quoteType', 'Stock'),  # EQUITY, ETF, etc.
                'beta': info.get('beta', 1)  # Default 1 si no hay beta disponible
            }
        except Exception as e:
            print(f"No se pudo obtener info fundamental para {simbolo}: {e}")
            return {
                'sector': 'N/A',
                'instrumento': 'N/A',
                'beta': 1  # Default 1 para evitar problemas en cálculos
            }

def main():
    # 1. Conexión a Google Sheets
    creds_dict = json.loads(GOOGLE_CREDS)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)
    
    # 2. Login en IOL
    iol = IOL_Manager()
    iol.login()
    print("✓ Login en IOL exitoso")
    
    # 3. Lógica de APPEND de Operaciones
    sheet_ops = spreadsheet.worksheet("operaciones")
    df_existente = pd.DataFrame(sheet_ops.get_all_records())
    ids_guardados = df_existente['id_operacion'].astype(str).tolist() if not df_existente.empty else []

    nuevas_ops = iol.get_operaciones_recientes()
    print(f"Operaciones recientes obtenidas de IOL: {len(nuevas_ops)}")
    
    rows_to_append = []
    
    for op in nuevas_ops:
        id_op = str(op.get('numero'))
        # Solo procesamos si es una operación de títulos y NO está ya en el Excel
        if id_op not in ids_guardados and op.get('tipo') in ['Compra', 'Venta']:
            cantidad = op.get('cantidad')
            # Si es venta, forzamos cantidad negativa para el neteo
            if op.get('tipo') == 'Venta':
                cantidad = -abs(cantidad)
            
            rows_to_append.append([
                op.get('fechaHora')[:10], # fecha
                op.get('simbolo'),        # simbolo
                op.get('tipo'),           # tipo
                cantidad,                 # cantidad
                op.get('precio'),         # precio_compra_venta
                'BCBA',                   # mercado (ajustar si operas afuera)
                id_op                     # id_operacion
            ])
    
    if rows_to_append:
        sheet_ops.append_rows(rows_to_append)
        print(f"✓ Se agregaron {len(rows_to_append)} operaciones nuevas.")
    else:
        print("No hay operaciones nuevas para agregar.")

    # 4. Actualización de COTIZACIONES
    # Leemos todos los símbolos que tenemos en operaciones para actualizar sus precios
    df_total_ops = pd.DataFrame(sheet_ops.get_all_records())
    print(f"Total de operaciones en el sheet: {len(df_total_ops)}")
    
    if df_total_ops.empty:
        print("⚠️ ADVERTENCIA: La hoja 'operaciones' está vacía. No hay símbolos para actualizar.")
        return
    
    # Ya no necesitamos la columna mercado
    simbolos = df_total_ops['simbolo'].drop_duplicates()
    print(f"Símbolos únicos a actualizar: {len(simbolos)}")
    print(f"Símbolos: {simbolos.tolist()}")
    
    cotizaciones_update = []
    maestra_update = []
    simbolos_no_encontrados = []
    
    for simbolo in simbolos:
        print(f"Consultando precio de {simbolo}...")
        
        # Primero intentar con IOL
        coti = iol.get_precio(simbolo)
        
        # Si no se encuentra en IOL, intentar con yfinance
        if not coti:
            print(f"  No encontrado en IOL, intentando con yfinance...")
            coti = iol.get_precio_yfinance(simbolo)
        
        if coti:
            print(f"✓ {simbolo}: ${coti.get('ultimoPrecio')}")
            cotizaciones_update.append([simbolo, coti['ultimoPrecio'], coti['fechaHora']])
            
            # Obtener información fundamental para la hoja maestra
            info_fund = iol.get_info_fundamental(simbolo)
            maestra_update.append([
                simbolo,
                info_fund['sector'],
                info_fund['instrumento'],
                info_fund['beta']  # Siempre será un número ahora
            ])
        else:
            print(f"✗ {simbolo}: No se obtuvo cotización (ignorado)")
            simbolos_no_encontrados.append(simbolo)
    
    print(f"\nTotal de cotizaciones obtenidas: {len(cotizaciones_update)}")
    if simbolos_no_encontrados:
        print(f"Símbolos no encontrados (ignorados): {simbolos_no_encontrados}")
    
    # Actualizar hoja de cotizaciones
    if cotizaciones_update:
        sheet_coti = spreadsheet.worksheet("cotizaciones")
        sheet_coti.clear()
        sheet_coti.update([['simbolo', 'ultimoPrecio', 'fecha_cotizacion']] + cotizaciones_update)
        print(f"✓ Precios actualizados en el sheet: {len(cotizaciones_update)} símbolos")
    else:
        print("⚠️ No se actualizó ningún precio (cotizaciones_update vacío)")
    
    # Actualizar hoja maestra
    if maestra_update:
        sheet_maestra = spreadsheet.worksheet("maestra")
        sheet_maestra.clear()
        sheet_maestra.update([['simbolo', 'sector', 'instrumento', 'beta']] + maestra_update)
        print(f"✓ Hoja maestra actualizada: {len(maestra_update)} símbolos")

if __name__ == "__main__":
    main()
