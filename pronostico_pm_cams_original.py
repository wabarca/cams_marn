# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:00:50 2021

@author: arw
"""

import os
import cdsapi
import zipfile
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date

#Cambio al directorio de trabajo
os.chdir('/home/arw/scripts/python/cams/temp')

import datetime
yesterday = datetime.date.fromordinal(datetime.date.today().toordinal()-1).strftime("%F")
string = f'{yesterday}/{yesterday}'

print(string)

#Script para analizar la 
#------------------------------------------------------------------>
c = cdsapi.Client()

c.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': [
            'dust_aerosol_optical_depth_550nm', 'particulate_matter_10um', 'particulate_matter_2.5um',
        ],
        #'date': '2022-10-23/2022-10-23',
        'date': string,
        'time': '12:00',
        'leadtime_hour': [
            '24', '25', '26',
            '27', '28', '29',
            '30', '31', '32',
            '33', '34', '35',
            '36', '37', '38',
            '39', '40', '41',
            '42', '43', '44',
            '45', '46', '47',
            '48', '49', '50',
            '51', '52', '53',
            '54', '55', '56',
            '57', '58', '59',
            '60', '61', '62',
            '63', '64', '65',
            '66', '67', '68',
            '69', '70', '71',
            '72', '73', '74',
            '75', '76', '77',
            '78', '79', '80',
            '81', '82', '83',
            '84', '85', '86',
            '87', '88', '89',
            '90',
        ],
        'type': 'forecast',
        'area': [
            16, -92, 12,
            -86,
        ],
        'format': 'netcdf_zip',
    },
    'download.netcdf_zip')
#------------------------------------------------------------------>
#Descomprime el archivo descargado
archivo_zip = zipfile.ZipFile("download.netcdf_zip")
archivo_zip.extractall()

#Leemos el netcdf que acabamos de bajar
nc = Dataset('data.nc')

#Borra archivos temporalaes
os.remove("download.netcdf_zip")
os.remove("data.nc")

#Inventario del contenido del archivo 1
#for key,value in nc.variables.items():
#    print(key)
#    print(value)
#    print()
    
    
#Lectura de latitudes y longitudes
lat = nc.variables['latitude'][:]
zlon = nc.variables['longitude'][:]    

#Lectura de niveles (Si los hubiere)
#nivel = nc.variables['level'][:]

#Construyendo el vector de tiempos
#Lectura y formato del tiempo
unidades = nc.variables['time'].units
calendario = nc.variables['time'].calendar
tiempo = nc.variables['time'][:]
tiempo = num2date(tiempo, units=unidades,calendar=calendario)
tiempo = [i.strftime("%d-%m-%Y %H:%M") for i in tiempo]

#Se setean los limites
levels_pm10 = np.arange(0.00000001,0.00000006,0.000000001)
levels_pm25 = np.arange(0.00000001,0.00000005,0.000000001)

#Lectura de variables, en este caso, contaminantes (Se omite el nivel porque no hay)
pm10 = nc.variables['pm10'][:,:,:]
pm25 = nc.variables['pm2p5'][:,:,:]
itime=len(tiempo)-1 #Asumo que el vector comienza en 0


#Plot PM10
for i in range (itime):
    #Seleccionamos el tiempo, para este caso no tenemos nivel
    #ilevel = 0
    
    variable = pm10[i,:,:]
    #variable = variable*10000000
    
    #Ahora realizamos un plot simple seleccionando un tiempo
    fig=plt.figure(figsize=(9,7), dpi=300)
    ax=fig.add_subplot(1,1,1,projection=ccrs.Mercator())
    
    #Esteticos
    ax.set_extent([-92,-86,12,16])
    ax.add_feature(cfeature.COASTLINE,lw=.5)
    #ax.add_feature(cfeature.RIVERS,lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.6)
    #Cabezera
    ax.set_title('Pronostico de PM 10 para %s' % tiempo[i])
    #Lineas
    ax.gridlines(xlocs=np.arange(-180,180,2.5),
    ylocs=np.arange(-90,90,2.5),
    draw_labels=True,color='gray',lw=0.1)
    pc=ax.contourf(zlon,lat,variable,
    levels=levels_pm10,
    transform=ccrs.PlateCarree(), 
    #cmap='twilight',)
    cmap='Reds',)
    cbar = fig.colorbar(pc, orientation="horizontal")
    
    
    fig.savefig("cams_pm10_"+f'{i}'+".png")
    plt.close(fig) 
    #plt.show()


#Plot PM25
for i in range (itime):
    #Seleccionamos el tiempo, para este caso no tenemos nivel
    #ilevel = 0
    
    variable = pm25[i,:,:]
    #variable = variable*10000000
    
    #Ahora realizamos un plot simple seleccionando un tiempo
    fig=plt.figure(figsize=(9,7), dpi=300)
    ax=fig.add_subplot(1,1,1,projection=ccrs.Mercator())
    
    #Esteticos
    ax.set_extent([-92,-86,12,16])
    ax.add_feature(cfeature.COASTLINE,lw=.5)
    #ax.add_feature(cfeature.RIVERS,lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.6)
    #Cabezera
    ax.set_title('Pronostico de PM 2.5 para %s' % tiempo[i])
    #Lineas
    ax.gridlines(xlocs=np.arange(-180,180,2.5),
    ylocs=np.arange(-90,90,2.5),
    draw_labels=True,color='gray',lw=0.1)
    pc=ax.contourf(zlon,lat,variable,
    levels=levels_pm25,
    transform=ccrs.PlateCarree(), 
    #cmap='twilight',)
    cmap='Reds',)
    cbar = fig.colorbar(pc, orientation="horizontal")
    
    fig.savefig("cams_pm25_"+f'{i}'+".png")
    plt.close(fig) 
