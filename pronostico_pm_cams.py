import os
import datetime
import zipfile
import numpy as np
import xarray as xr
import cdsapi
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as image
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuración inicial
WORKDIR = "/home/arw/scripts/python/cams/temp"
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

# Descarga de AOD para el área extendida [-100, 0] x [0, 30]
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

# Lectura de datos
ds_sfc = xr.open_dataset("data_sfc_polvo.nc")
ds_plev = xr.open_dataset("data_plev_polvo.nc")
ds_sfc_aod = xr.open_dataset("data_sfc_aod.nc")
aod = ds_sfc_aod["duaod550"].values.squeeze()
tiempo_sfc_aod = ds_sfc_aod.forecast_reference_time.values[0] + ds_sfc_aod.forecast_period.values
tiempo_sfc_aod_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_sfc_aod]
lat_aod = ds_sfc_aod.latitude.values
lon_aod = ds_sfc_aod.longitude.values
X_aod, Y_aod = np.meshgrid(lon_aod, lat_aod)

lat = ds_sfc.latitude.values
lon = ds_sfc.longitude.values
X, Y = np.meshgrid(lon, lat)

# Reconstrucción de los tiempos reales
tiempo_sfc = ds_sfc.forecast_reference_time.values[0] + ds_sfc.forecast_period.values
tiempo_plev = ds_plev.forecast_reference_time.values[0] + ds_plev.forecast_period.values

tiempo_sfc_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_sfc]
tiempo_plev_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_plev]

# Variables
pm10 = ds_sfc.pm10.values.squeeze() * 1e9
pm25 = ds_sfc.pm2p5.values.squeeze() * 1e9
rho_aire = 1.225
dust_total = (
    ds_plev["aermr04"].sel(pressure_level=1000).values +
    ds_plev["aermr05"].sel(pressure_level=1000).values +
    ds_plev["aermr06"].sel(pressure_level=1000).values
).squeeze() * rho_aire * 1e9

# Shapefiles y logos
shp1 = gpd.read_file("/home/arw/shape/GSHHS_h_L1.shp")
shp2 = gpd.read_file("/home/arw/shape/ESA_CA_wgs84.shp")
shp3 = gpd.read_file("/home/arw/shape/El_Salvador_departamentos.shp")
logo = image.imread("/home/arw/scripts/python/cams/logoMarn_color.png")
icca = image.imread("/home/arw/scripts/python/cams/ICCA.jpeg")

# Función de graficado con proyección y ajustes visuales
def graficar_variable(variable, tiempos, X, Y, lat, lon, logo, etiqueta_hora, cmap, niveles,
                      nombre_variable, nombre_archivo_base, icca=None, niveles_icca=None,
                      categorias=None, usar_icca=False, shapefiles=[]):
    aspect_ratio = (max(lon) - min(lon)) / (max(lat) - min(lat))
    width = 12
    height = width / aspect_ratio

    for i in range(min(variable.shape[0], len(tiempos))):
        fig = plt.figure(figsize=(width, height), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree())

        cont = ax.contourf(X, Y, variable[i], levels=niveles,
                           cmap=cmap if not usar_icca else None,
                           colors=cmap if usar_icca else None,
                           extend="both", transform=ccrs.PlateCarree())

        cbar = plt.colorbar(cont, ax=ax, orientation='horizontal', pad=0.07, shrink=0.4, extendrect=True)
        cbar.outline.set_linewidth(0.5)
        if usar_icca and categorias:
            cont.set_clim(min(niveles_icca), max(niveles_icca))
            cbar.set_ticks(niveles_icca)
            cbar.set_ticklabels(categorias)

        ax.set_title(f"{nombre_variable} - {tiempos[i]}\nModelo CAMS - Observatorio de Amenazas - MARN", fontsize=13, pad=15)
        ax.set_xlabel("Longitud", fontsize=11)
        ax.set_ylabel("Latitud", fontsize=11)

        gl = ax.gridlines(draw_labels=True)
        gl.linewidth = 0.5
        gl.linestyle = '--'
        gl.color = 'gray'

        for shp in shapefiles:
            shp.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())

        # Logo
        logo_height = (max(lat) - min(lat)) * 0.12
        logo_width = (logo.shape[1] / logo.shape[0]) * logo_height
        ax.imshow(logo, extent=[max(lon)-logo_width, max(lon), min(lat), min(lat)+logo_height],
                  transform=ccrs.PlateCarree(), zorder=10)

        # ICCA leyenda en la esquina inferior izquierda
        if usar_icca and icca is not None:
            icca_height = (max(lat) - min(lat)) * 0.12
            icca_width = (icca.shape[1] / icca.shape[0]) * icca_height
            ax.imshow(icca, extent=[min(lon), min(lon)+icca_width, min(lat), min(lat)+icca_height],
                      transform=ccrs.PlateCarree(), zorder=10)

        # Hora de creación como pie de página
        fig.text(0.5, 0.01, etiqueta_hora, fontsize=8, ha='center')

        plt.tight_layout()
        fig.savefig(f"{nombre_archivo_base}_{i}.png", bbox_inches='tight')
        plt.close()

# Parámetros
niveles_pm10 = np.arange(0, 200, 1)
niveles_pm25 = np.arange(0, 100, 1)
niveles_dust = np.arange(0, 300, 10)
niveles_aod = np.arange(0, 1.1, 0.1)

paleta_icca = ["#92d14f", "#ffff01", "#ffc000", "#fe0000", "#7030a0", "#000000"]
niveles_pm10_icca = [56, 155, 255, 355, 424, 604]
niveles_pm25_icca = [15.5, 40.5, 66, 160, 251, 500]
categorias = ["Buena", "Moderada", "Dañina sensibles", "Dañina salud", "Muy dañina", "Peligroso"]

# Ejecutar
graficar_variable(pm10, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora, "YlOrBr", niveles_pm10,
                  "PM10 (µg/m³)", "cams_pm10", shapefiles=[shp1, shp2, shp3])

graficar_variable(pm25, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora, "YlOrBr", niveles_pm25,
                  "PM2.5 (µg/m³)", "cams_pm25", shapefiles=[shp1, shp2, shp3])

graficar_variable(pm10, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora, paleta_icca, niveles_pm10_icca,
                  "PM10 ICCA", "cams_pm10_icca", icca=icca, niveles_icca=niveles_pm10_icca,
                  categorias=categorias, usar_icca=True, shapefiles=[shp1, shp2, shp3])

graficar_variable(pm25, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora, paleta_icca, niveles_pm25_icca,
                  "PM2.5 ICCA", "cams_pm25_icca", icca=icca, niveles_icca=niveles_pm25_icca,
                  categorias=categorias, usar_icca=True, shapefiles=[shp1, shp2, shp3])

graficar_variable(dust_total, tiempo_plev_str, X, Y, lat, lon, logo, etiqueta_hora, "YlOrBr", niveles_dust,
                  "Concentración de polvo (µg/m³)", "cams_dust_total", shapefiles=[shp1, shp2, shp3])

graficar_variable(aod, tiempo_sfc_aod_str, X_aod, Y_aod, lat_aod, lon_aod, logo, etiqueta_hora, "YlOrBr",
                  niveles_aod, "AOD polvo 550nm", "cams_aod_dust", shapefiles=[shp1, shp2])
