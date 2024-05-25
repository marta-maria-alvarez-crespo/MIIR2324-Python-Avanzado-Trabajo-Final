import cProfile
import pstats
import pandas as pd
from io import StringIO

# Define tu función principal
def main():
    # Tu código aquí
    pass

# Ejecuta cProfile y captura la salida en un buffer
profile_buffer = StringIO()
cProfile.run('main()', filename=None, sort='cumulative', stream=profile_buffer)

# Vuelve al inicio del buffer
profile_buffer.seek(0)

# Usa pstats para leer los datos del buffer
stats = pstats.Stats(profile_buffer)

# Extrae los datos y conviértelos a un formato de lista de listas
stats_data = []
for func, (cc, nc, tt, ct, callers) in stats.stats.items():
    filename, line, funcname = func
    stats_data.append({
        'filename': filename,
        'line': line,
        'funcname': funcname,
        'cc': cc,
        'nc': nc,
        'tt': tt,
        'ct': ct
    })

# Convierte los datos en un DataFrame de pandas
df = pd.DataFrame(stats_data)

# Muestra el DataFrame
print(df)
