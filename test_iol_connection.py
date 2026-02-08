"""
Script de prueba para testear la conexión con la API de IOL
Ejecuta este script primero para verificar que todo funciona
"""

import requests
import json

# ===================================
# CONFIGURACIÓN
# ===================================
IOL_USERNAME = "milosmaggi@gmail.com"  # Reemplazar con tu usuario
IOL_PASSWORD = "Ramonlina0812$"  # Reemplazar con tu contraseña

def test_autenticacion():
    """Prueba la autenticación con IOL"""
    print("🔐 Probando autenticación con IOL...")
    
    url = "https://api.invertironline.com/token"
    
    data = {
        'username': IOL_USERNAME,
        'password': IOL_PASSWORD,
        'grant_type': 'password'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        response = requests.post(url, data=data, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Autenticación exitosa!")
            print(f"📝 Bearer Token (primeros 20 caracteres): {token_data['access_token'][:20]}...")
            print(f"🔄 Refresh Token (primeros 20 caracteres): {token_data['refresh_token'][:20]}...")
            return token_data['access_token']
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")
        return None


def test_cotizacion(bearer_token, simbolo="GGAL"):
    """Prueba obtener la cotización de un título"""
    print(f"\n📊 Probando obtener cotización de {simbolo}...")
    
    url = f"https://api.invertironline.com/api/v2/bCBA/Titulos/{simbolo}/Cotizacion"
    
    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Cotización obtenida!")
            print(f"📈 Último precio: ${data.get('ultimoPrecio', 'N/A')}")
            print(f"📅 Fecha: {data.get('fechaHora', 'N/A')}")
            print(f"🔼 Variación: {data.get('variacionPorcentual', 'N/A')}%")
            
            # Mostrar toda la respuesta para debugging
            print("\n📋 Respuesta completa:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")
        return None


def test_portafolio(bearer_token):
    """Prueba obtener el portafolio"""
    print("\n💼 Probando obtener portafolio...")
    
    url = "https://api.invertironline.com/api/v2/portafolio/argentina"
    
    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Portafolio obtenido!")
            
            # Mostrar activos
            activos = data.get('activos', [])
            print(f"\n📊 Tienes {len(activos)} activos en tu portafolio:")
            
            for activo in activos[:5]:  # Mostrar solo los primeros 5
                print(f"  • {activo.get('titulo', {}).get('simbolo', 'N/A')}: "
                      f"{activo.get('cantidad', 0)} unidades - "
                      f"Valor: ${activo.get('valorizado', 0):,.2f}")
            
            if len(activos) > 5:
                print(f"  ... y {len(activos) - 5} más")
                
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")
        return None


def main():
    print("=" * 60)
    print("🧪 TEST DE CONEXIÓN CON API DE INVERTIR ONLINE")
    print("=" * 60)
    
    # Test 1: Autenticación
    bearer_token = test_autenticacion()
    
    if not bearer_token:
        print("\n⛔ No se pudo autenticar. Verifica tus credenciales.")
        return
    
    # Test 2: Obtener cotización
    test_cotizacion(bearer_token, "GGAL")
    
    # Test 3: Obtener portafolio
    test_portafolio(bearer_token)
    
    print("\n" + "=" * 60)
    print("✅ Tests completados!")
    print("=" * 60)


if __name__ == "__main__":
    main()