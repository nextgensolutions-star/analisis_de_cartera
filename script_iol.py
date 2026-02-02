import os
import json
import requests
import gspread
import pandas as pd
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

    def get_precio(self, simbolo, mercado="BCBA"):
        url = f"{self.url_base}/api/v1/Titulos/{mercado}/Instrumentos/{simbolo}/Cotizacion"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                return r.json()
        except: return None
        return None

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
    
    # 3. Lógica de APPEND de Operaciones
    sheet_ops = spreadsheet.worksheet("operaciones")
    df_existente = pd.DataFrame(sheet_ops.get_all_records())
    ids_guardados = df_existente['id_operacion'].astype(str).tolist() if not df_existente.empty else []

    nuevas_ops = iol.get_operaciones_recientes()
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
        print(f"Se agregaron {len(rows_to_append)} operaciones nuevas.")

    # 4. Actualización de COTIZACIONES
    # Leemos todos los símbolos que tenemos en operaciones para actualizar sus precios
    df_total_ops = pd.DataFrame(sheet_ops.get_all_records())
    simbolos = df_total_ops[['simbolo', 'mercado']].drop_duplicates()
    
    cotizaciones_update = []
    for _, row in simbolos.iterrows():
        coti = iol.get_precio(row['simbolo'], row['mercado'])
        if coti:
            cotizaciones_update.append([row['simbolo'], coti['ultimoPrecio'], coti['fechaHora']])
    
    sheet_coti = spreadsheet.worksheet("cotizaciones")
    sheet_coti.clear()
    sheet_coti.update([['simbolo', 'ultimoPrecio', 'fecha_cotizacion']] + cotizaciones_update)
    print("Precios actualizados.")

if __name__ == "__main__":
    main()