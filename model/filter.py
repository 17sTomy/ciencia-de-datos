import pandas as pd
import os

# --- 1. Conversión y Unificación en un solo paso eficiente ---


FILES = [] 
FINAL_PARQUET_PATH = "model/filter/all_data.parquet"

# Eliminamos el archivo final si ya existe para empezar de cero
if os.path.exists(FINAL_PARQUET_PATH):
    os.remove(FINAL_PARQUET_PATH)

print("Iniciando la conversión y unificación de archivos SAS a Parquet...")

# Procesamos el primer archivo por separado para crear el archivo base
try:
    print(f"Procesando archivo base: {FILES[0]}...")
    df_first = pd.read_sas(f"model/input/{FILES[0]}", format="sas7bdat", encoding="latin1")
    # Escribimos el primer archivo en modo 'overwrite' (el modo por defecto)
    df_first.to_parquet(FINAL_PARQUET_PATH, engine="fastparquet", index=False)
    # Liberamos la memoria
    del df_first 
    print(f"Archivo base '{FILES[0]}' procesado y guardado.")

except Exception as e:
    print(f"Error procesando el primer archivo {FILES[0]}: {e}")
    # Si el primer archivo falla, no continuamos
    exit()

# Procesamos el resto de los archivos en un bucle, añadiéndolos al archivo existente
# Empezamos desde el segundo archivo (índice 1)
for file in FILES[1:]:
    try:
        print(f"Procesando y añadiendo archivo: {file}...")
        path = f"model/input/{file}"
        
        # Leemos un archivo SAS a la vez
        df_temp = pd.read_sas(path, format="sas7bdat", encoding="latin1")
        
        # ¡LA CLAVE! Añadimos los datos al archivo Parquet existente
        # en lugar de sobreescribirlo.
        df_temp.to_parquet(FINAL_PARQUET_PATH, index=False,engine="fastparquet", append=True)
        
        # Liberamos la memoria inmediatamente
        del df_temp 
        print(f"Archivo '{file}' añadido correctamente.")
        
    except Exception as e:
        print(f"Error procesando el archivo {file}: {e}")
        # Puedes decidir si continuar con los demás archivos o parar
        continue

print(f"\nProceso completado. Todos los datos están unificados en '{FINAL_PARQUET_PATH}'.")