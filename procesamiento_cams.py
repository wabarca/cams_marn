import os
import datetime
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as image
import cartopy.crs as ccrs

# === Directorios ===
DATA_DIR = "/home/arw/cams/temp/"
IMG_DIR = "/home/arw/cams/imagery/"

# Crear carpeta y limpiar imágenes anteriores
os.makedirs(IMG_DIR, exist_ok=True)
for f in os.listdir(IMG_DIR):
    if f.startswith("cams_") and f.endswith(".png"):
        os.remove(os.path.join(IMG_DIR, f))

# === Lectura de archivos .nc ===
ds_sfc = xr.open_dataset(os.path.join(DATA_DIR, "data_sfc_polvo.nc"))
ds_plev = xr.open_dataset(os.path.join(DATA_DIR, "data_plev_polvo.nc"))
ds_sfc_aod = xr.open_dataset(os.path.join(DATA_DIR, "data_sfc_aod.nc"))

# === Procesamiento de tiempos ===
aod = ds_sfc_aod["duaod550"].values.squeeze()
tiempo_sfc_aod = ds_sfc_aod.forecast_reference_time.values[0] + ds_sfc_aod.forecast_period.values
tiempo_sfc_aod_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_sfc_aod]

lat_aod = ds_sfc_aod.latitude.values
lon_aod = ds_sfc_aod.longitude.values
X_aod, Y_aod = np.meshgrid(lon_aod, lat_aod)

lat = ds_sfc.latitude.values
lon = ds_sfc.longitude.values
X, Y = np.meshgrid(lon, lat)

tiempo_sfc = ds_sfc.forecast_reference_time.values[0] + ds_sfc.forecast_period.values
tiempo_plev = ds_plev.forecast_reference_time.values[0] + ds_plev.forecast_period.values

tiempo_sfc_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_sfc]
tiempo_plev_str = [np.datetime_as_string(t - np.timedelta64(6, 'h'), unit='m') for t in tiempo_plev]

# === Variables ===
pm10 = ds_sfc.pm10.values.squeeze() * 1e9
pm25 = ds_sfc.pm2p5.values.squeeze() * 1e9
rho_aire = 1.225
dust_total = (
    ds_plev["aermr04"].sel(pressure_level=1000).values +
    ds_plev["aermr05"].sel(pressure_level=1000).values +
    ds_plev["aermr06"].sel(pressure_level=1000).values
).squeeze() * rho_aire * 1e9

# === Shapefiles y logos ===
shp1 = gpd.read_file("/home/arw/shape/GSHHS_h_L1.shp")
shp2 = gpd.read_file("/home/arw/shape/ESA_CA_wgs84.shp")
shp3 = gpd.read_file("/home/arw/shape/El_Salvador_departamentos.shp")
logo = image.imread("/home/arw/scripts/python/cams/logoMarn_color.png")
icca = image.imread("/home/arw/scripts/python/cams/ICCA.jpeg")

# === Parámetros ===
niveles_pm10 = np.arange(0, 200, 1)
niveles_pm25 = np.arange(0, 100, 1)
niveles_dust = np.arange(0, 100, 10)
niveles_aod = np.arange(0, 1.1, 0.1)

paleta_icca = ["#92d14f", "#ffff01", "#ffc000", "#fe0000", "#7030a0", "#000000"]
niveles_pm10_icca = [0, 56, 155, 255, 355, 424, 604]
niveles_pm25_icca = [0, 15.5, 40.5, 66, 160, 251, 500]
categorias = ["Buena", "Moderada", "Dañina\n sensibles", "Dañina\n salud", "Muy\n dañina", "Peligroso"]

etiqueta_hora = f"Hora de creación: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} (hora local)"

# === Función de graficado ===
def graficar_variable(variable, tiempos, X, Y, lat, lon, logo, etiqueta_hora, cmap, niveles,
                      nombre_variable, nombre_archivo_base, icca=None, niveles_icca=None,
                      categorias=None, usar_icca=False, shapefiles=[], shrink_colorbar=0.4):
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

        cbar = plt.colorbar(cont, ax=ax, orientation='horizontal', pad=0.08, shrink=shrink_colorbar, extendrect=True)
        cbar.outline.set_linewidth(0.5)
        if usar_icca and categorias:
            cont.set_clim(min(niveles_icca), max(niveles_icca))
            ticks_icca = [(niveles_icca[i] + niveles_icca[i+1]) / 2 for i in range(len(niveles_icca)-1)]
            cbar.set_ticks(ticks_icca)
            cbar.set_ticklabels(categorias)

        ax.set_title(f"{nombre_variable} - {tiempos[i]} (hora local)\nModelo CAMS - Observatorio de Amenazas - MARN", fontsize=12, pad=15)
        ax.set_xlabel("Longitud", fontsize=11)
        ax.set_ylabel("Latitud", fontsize=11)

        gl = ax.gridlines(draw_labels=True)
        gl.linewidth = 0.5
        gl.linestyle = '--'
        gl.color = 'gray'

        for shp in shapefiles:
            shp.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())

        logo_height = (max(lat) - min(lat)) * 0.12
        logo_width = (logo.shape[1] / logo.shape[0]) * logo_height
        ax.imshow(logo, extent=[max(lon)-logo_width, max(lon), min(lat), min(lat)+logo_height],
                  transform=ccrs.PlateCarree(), zorder=10)

        if usar_icca and icca is not None:
            icca_height = (max(lat) - min(lat)) * 0.4
            icca_width = (icca.shape[1] / icca.shape[0]) * icca_height
            ax.imshow(icca, extent=[min(lon), min(lon)+icca_width, min(lat), min(lat)+icca_height],
                      transform=ccrs.PlateCarree(), zorder=10)

        fig.text(0.5, 0.01, etiqueta_hora, fontsize=7, ha='center')
        plt.tight_layout()
        fig.savefig(f"{nombre_archivo_base}_{i+1}.png", bbox_inches='tight')
        plt.close()

# === Ejecuciones de graficado ===
graficar_variable(aod, tiempo_sfc_aod_str, X_aod, Y_aod, lat_aod, lon_aod, logo, etiqueta_hora,
                  "YlOrBr", niveles_aod, "AOD polvo 550nm",
                  os.path.join(IMG_DIR, "cams_aod_dust"),
                  shapefiles=[shp1, shp2], shrink_colorbar=0.25)

graficar_variable(pm10, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora,
                  paleta_icca, niveles_pm10_icca, "PM10 ICCA",
                  os.path.join(IMG_DIR, "cams_pm10_icca"),
                  icca=icca, niveles_icca=niveles_pm10_icca, categorias=categorias,
                  usar_icca=True, shapefiles=[shp1, shp2, shp3])

graficar_variable(pm25, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora,
                  paleta_icca, niveles_pm25_icca, "PM2.5 ICCA",
                  os.path.join(IMG_DIR, "cams_pm25_icca"),
                  icca=icca, niveles_icca=niveles_pm25_icca, categorias=categorias,
                  usar_icca=True, shapefiles=[shp1, shp2, shp3])

graficar_variable(pm10, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora,
                  "YlOrBr", niveles_pm10, "PM10 (µg/m³)",
                  os.path.join(IMG_DIR, "cams_pm10"),
                  shapefiles=[shp1, shp2, shp3])

graficar_variable(pm25, tiempo_sfc_str, X, Y, lat, lon, logo, etiqueta_hora,
                  "YlOrBr", niveles_pm25, "PM2.5 (µg/m³)",
                  os.path.join(IMG_DIR, "cams_pm25"),
                  shapefiles=[shp1, shp2, shp3])

graficar_variable(dust_total, tiempo_plev_str, X, Y, lat, lon, logo, etiqueta_hora,
                  "YlOrBr", niveles_dust, "Concentración de polvo (µg/m³)",
                  os.path.join(IMG_DIR, "cams_dust_total"),
                  shapefiles=[shp1, shp2, shp3])
