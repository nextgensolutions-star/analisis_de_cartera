import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import time

# ===================================
# CONFIGURACIÓN IOL API
# ===================================
class IOLClient:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "https://api.invertironline.com"
        self.bearer_token = None
        self.refresh_token = None
        self.token_expiry = None
        
    def authenticate(self):
        """Obtiene el bearer token inicial"""
        url = f"{self.base_url}/token"
        
        data = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password'
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.post(url, data=data, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            self.bearer_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            self.token_expiry = time.time() + 900  # 15 minutos
            print("✅ Autenticación exitosa")
            return True
        else:
            print(f"❌ Error en autenticación: {response.status_code}")
            print(response.text)
            return False
    
    def refresh_bearer_token(self):
        """Refresca el bearer token usando el refresh token"""
        url = f"{self.base_url}/token"
        
        data = {
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = requests.post(url, data=data, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            self.bearer_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            self.token_expiry = time.time() + 900
            print("🔄 Token refrescado")
            return True
        else:
            print(f"❌ Error al refrescar token: {response.status_code}")
            return False
    
    def check_token(self):
        """Verifica si el token está por expirar y lo refresca si es necesario"""
        if time.time() > self.token_expiry - 60:  # Refresca 1 min antes
            return self.refresh_bearer_token()
        return True
    
    def get_cotizacion(self, simbolo, mercado="bCBA", pais="argentina"):
        """
        Obtiene la cotización de un título
        
        Args:
            simbolo: Símbolo del título (ej: GGAL, YPFD)
            mercado: Mercado (bCBA por defecto)
            pais: País (argentina por defecto)
        """
        self.check_token()
        
        # Endpoint: GET /api/v2/{mercado}/Titulos/{simbolo}/Cotizacion
        url = f"{self.base_url}/api/v2/{mercado}/Titulos/{simbolo}/Cotizacion"
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # La respuesta contiene el precio en diferentes campos según el mercado
                # Generalmente: ultimoPrecio, puntas.precioCompra, puntas.precioVenta
                ultimo_precio = data.get('ultimoPrecio', 0)
                return ultimo_precio
            else:
                print(f"⚠️ Error al obtener cotización de {simbolo}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Excepción al consultar {simbolo}: {str(e)}")
            return None
    
    def get_precio_from_panel(self, simbolo, instrumento="acciones", panel="accionesLideres", pais="argentina"):
        """
        Obtiene cotización usando el endpoint de paneles
        Endpoint: GET /api/v2/Cotizaciones/{Instrumento}/{Panel}/{Pais}
        """
        self.check_token()
        
        url = f"{self.base_url}/api/v2/Cotizaciones/{instrumento}/{panel}/{pais}"
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # Buscar el símbolo en la lista de títulos
                titulos = data.get('titulos', [])
                for titulo in titulos:
                    if titulo.get('simbolo') == simbolo:
                        return titulo.get('ultimoPrecio', 0)
                return None
            else:
                print(f"⚠️ Error al obtener panel: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Excepción en panel: {str(e)}")
            return None


# ===================================
# GOOGLE SHEETS
# ===================================
class GoogleSheetsUpdater:
    def __init__(self, credentials_file, spreadsheet_name):
        """
        Inicializa la conexión con Google Sheets
        
        Args:
            credentials_file: Ruta al archivo JSON de credenciales de servicio
            spreadsheet_name: Nombre del Google Sheet
        """
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
        self.client = gspread.authorize(creds)
        self.sheet = self.client.open(spreadsheet_name).sheet1  # Primera hoja
        
    def get_operaciones(self):
        """Lee todas las operaciones del sheet"""
        records = self.sheet.get_all_records()
        return records
    
    def update_precio(self, row_index, precio, detalle="", fecha=None):
        """
        Actualiza el precio en una fila específica
        
        Args:
            row_index: Índice de la fila (1-based, incluye header)
            precio: Precio a actualizar (puede ser None)
            detalle: Comentario o mensaje de error
            fecha: Fecha de la consulta (opcional)
        """
        if fecha is None:
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Columna H = ultimoPrecio (columna 8)
        # Columna I = detalle (columna 9)
        
        try:
            if precio is not None:
                self.sheet.update_cell(row_index, 8, precio)
                print(f"✅ Fila {row_index}: ${precio:,.2f}")
            
            # Siempre actualizar detalle
            if detalle:
                self.sheet.update_cell(row_index, 9, detalle)
                if precio is None:
                    print(f"⚠️ Fila {row_index}: {detalle}")
                else:
                    print(f"📝 Detalle: {detalle}")
        except Exception as e:
            print(f"❌ Error actualizando fila {row_index}: {str(e)}")
    
    def ensure_detalle_column(self):
        """Verifica que exista la columna 'detalle' en el header, si no la crea"""
        try:
            headers = self.sheet.row_values(1)  # Primera fila (headers)
            
            # Si no hay columna I o está vacía, agregar header "detalle"
            if len(headers) < 9 or not headers[8]:
                self.sheet.update_cell(1, 9, "detalle")
                print("✅ Columna 'detalle' creada en columna I")
            else:
                print(f"✅ Columna I ya existe: '{headers[8]}'")
        except Exception as e:
            print(f"⚠️ No se pudo verificar columna detalle: {str(e)}")


# ===================================
# FUNCIÓN PRINCIPAL
# ===================================
def main():
    # CONFIGURACIÓN - Lee desde variables de entorno (para GitHub Actions)
    # o usa valores por defecto (para ejecución local)
    import os
    
    IOL_USERNAME = os.getenv("IOL_USERNAME", "TU_USUARIO_IOL")
    IOL_PASSWORD = os.getenv("IOL_PASSWORD", "TU_CONTRASEÑA_IOL")
    GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE", "credenciales_google.json")
    SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "operaciones")
    
    print("🚀 Iniciando actualización de precios desde IOL...")
    
    # 1. Autenticar con IOL
    iol = IOLClient(IOL_USERNAME, IOL_PASSWORD)
    if not iol.authenticate():
        print("❌ No se pudo autenticar con IOL")
        return
    
    # 2. Conectar con Google Sheets
    try:
        gs = GoogleSheetsUpdater(GOOGLE_CREDS_FILE, SPREADSHEET_NAME)
        print("✅ Conectado a Google Sheets")
        
        # Verificar/crear columna detalle
        gs.ensure_detalle_column()
    except Exception as e:
        print(f"❌ Error al conectar con Google Sheets: {str(e)}")
        return
    
    # 3. Obtener operaciones
    operaciones = gs.get_operaciones()
    print(f"📊 Se encontraron {len(operaciones)} operaciones")
    
    # 4. Actualizar precios
    fecha_consulta = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for idx, op in enumerate(operaciones, start=2):  # start=2 porque fila 1 es header
        simbolo = op.get('simbolo', '')
        mercado = op.get('mercado', 'BCBA')
        
        if not simbolo:
            detalle = "Sin símbolo especificado"
            gs.update_precio(idx, None, detalle, fecha_consulta)
            continue
        
        print(f"\n🔍 Consultando {simbolo} en {mercado}...")
        
        # Solo consultar si es del mercado BCBA (IOL no tiene crypto)
        if mercado == 'BCBA':
            precio = iol.get_cotizacion(simbolo, mercado="bCBA")
            
            # Si falla, intentar con el endpoint de paneles
            if precio is None:
                precio = iol.get_precio_from_panel(simbolo)
            
            if precio and precio > 0:
                detalle = f"OK - Actualizado {fecha_consulta}"
                gs.update_precio(idx, precio, detalle, fecha_consulta)
            else:
                detalle = f"Ticker '{simbolo}' no encontrado en IOL"
                gs.update_precio(idx, None, detalle, fecha_consulta)
                print(f"⚠️ No se pudo obtener precio para {simbolo}")
        elif mercado == 'Binance':
            detalle = "Mercado Binance - No disponible en IOL"
            gs.update_precio(idx, None, detalle, fecha_consulta)
            print(f"⏭️ {simbolo} es de {mercado}, no disponible en IOL")
        else:
            detalle = f"Mercado '{mercado}' - No disponible en IOL"
            gs.update_precio(idx, None, detalle, fecha_consulta)
            print(f"⏭️ {simbolo} es de {mercado}, saltando (IOL solo tiene BCBA)")
        
        # Pausa para no saturar la API
        time.sleep(0.5)
    
    print("\n✅ Actualización completada!")


if __name__ == "__main__":
    main()