# -*- coding: utf-8 -*-
import os
import datetime
import zipfile
import cdsapi

# Configuración inicial
WORKDIR = "/var/www/html/cams/"
os.chdir(WORKDIR)

# Eliminar archivos .zip y .nc previos
for f in os.listdir(WORKDIR):
    if f.endswith(".zip") or f.endswith(".nc"):
        os.remove(os.path.join(WORKDIR, f))

# Fecha de ayer
yesterday = datetime.date.today() - datetime.timedelta(days=1)
fecha_str = yesterday.strftime("%Y-%m-%d")
print(f"Fecha de ayer: {fecha_str}")
etiqueta_hora = f"Hora de creación: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} Hora local"

# Descarga de PM10, PM2.5 y concentraciones de polvo (área Centroamérica)
client = cdsapi.Client()
client.retrieve(
    "cams-global-atmospheric-composition-forecasts",
    {
        "variable": [
            "particulate_matter_2.5um",
            "particulate_matter_10um",
            "dust_aerosol_0.03-0.55um_mixing_ratio",
            "dust_aerosol_0.55-0.9um_mixing_ratio",
            "dust_aerosol_0.9-20um_mixing_ratio"
        ],
        "pressure_level": ["1000"],
        "date": f"{fecha_str}/{fecha_str}",
        "time": ["00:00"],
        "leadtime_hour": [str(i) for i in range(121)],
        "type": "forecast",
        "data_format": "netcdf_zip",
        "area": [17, -93, 11, -82.33],
    }
).download("data_polvo.zip")

# Descarga de AOD para el área extendida (lon[-100,0], lat[0,30])
# Orden: [north, west, south, east] = [30, -100, 0, 0]
client.retrieve(
    "cams-global-atmospheric-composition-forecasts",
    {
        "variable": ["dust_aerosol_optical_depth_550nm"],
        "date": f"{fecha_str}/{fecha_str}",
        "time": ["00:00"],
        "leadtime_hour": [str(i) for i in range(121)],
        "type": "forecast",
        "data_format": "netcdf_zip",
        "area": [30, -100, 0, 0],
    }
).download("data_aod.zip")

# Descomprimir y renombrar archivos NetCDF
def descomprimir_y_renombrar(nombre_zip, sufijo):
    with zipfile.ZipFile(nombre_zip, "r") as zip_ref:
        file_list = zip_ref.namelist()
        for name in file_list:
            zip_ref.extract(name, WORKDIR)
            if "plev" in name:
                os.rename(os.path.join(WORKDIR, name), os.path.join(WORKDIR, f"data_plev_{sufijo}.nc"))
            elif "sfc" in name:
                os.rename(os.path.join(WORKDIR, name), os.path.join(WORKDIR, f"data_sfc_{sufijo}.nc"))

descomprimir_y_renombrar("data_polvo.zip", "polvo")
descomprimir_y_renombrar("data_aod.zip", "aod")

# Eliminar archivos .zip y .nc previos
for f in os.listdir(WORKDIR):
    if f.endswith(".zip"):
        os.remove(os.path.join(WORKDIR, f))
