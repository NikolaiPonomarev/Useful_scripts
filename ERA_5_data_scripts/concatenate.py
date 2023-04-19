import cdsapi
import datetime
import numpy as np
from multiprocessing import Pool
from multiprocessing import Pool
import xarray as xr
import  os, sys

#
#This code converts era5 data from grid to netcdf format and also handles the conversion of 
#the volumetric soil moisture to the soil moisture index (SMI) required for ICON simulations

# Generate lists of dates and times of the day
start_day = datetime.datetime.strptime("2022-12-23", "%Y-%m-%d")
end_day = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")

generated_dates = [start_day + datetime.timedelta(days=x) for x in range(0, (end_day-start_day).days+1)]

start = datetime.datetime.strptime("2022-01-01:00:00", "%Y-%m-%d:%H:%M")
end = datetime.datetime.strptime("2022-01-01:23:00", "%Y-%m-%d:%H:%M")
generated_times = np.array([datetime.time(i,0) for i in range(start.hour,end.hour+1)])

print('Given dates ', generated_dates, generated_times)


times_for_each_day = np.repeat([np.array(generated_times)], len(generated_dates), axis=0).flatten()
dates_for_each_time = np.repeat(generated_dates, len(generated_times))

def merge_gribs(date, time):
    print(date, time)
    inputfile_ml = 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.grib'
    inputfile_sf = 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.grib'

    outputfile_ml = 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.nc'
    outputfile_sf = 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.nc'

    #use cdo to convert from grib to nc
    os.system('cdo -t ecmwf -f nc copy ' + inputfile_ml + ' ' + outputfile_ml)
    os.system('cdo -t ecmwf -f nc copy ' + inputfile_sf + ' ' + outputfile_sf)

    ds_ml = xr.open_dataset(outputfile_ml)
    ds_surf = xr.open_dataset(outputfile_sf)

    #convert soil volumetric moisture to soil moisture index using 7 soil types
    wiltingp = [0.059, 0.151, 0.133, 0.279, 0.335, 0.267, 0.151]  # wilting point
    fieldcap =  [0.242, 0.346, 0.382, 0.448, 0.541, 0.662, 0.346]  # field capacity

    soil_layers = ['SWVL1', 'SWVL2', 'SWVL3', 'SWVL4']
    sl_t = getattr(ds_surf, 'SLT').values[0]
    
    


    for layer in soil_layers:

        swvl = getattr(ds_surf, layer).values[0]


        mask = (sl_t == 0+1).astype(int)
        #print('SWVL',np.shape(swvl), np.shape(mask), np.min(sl_t), np.max(sl_t))
        sl = (swvl-wiltingp[0])/(fieldcap[0]-wiltingp[0])*mask
        #print('SL', np.sum(sl.flatten()))
        for slt in range(1, 7):
            mask = (sl_t == slt+1).astype(int)
            sl = sl+(swvl-wiltingp[slt])/(fieldcap[slt]-wiltingp[slt])*mask
            #print('SL', np.sum(sl.flatten()), slt, wiltingp[slt])
        ds_surf[layer].values[0] = sl  

    
    ds_merged = xr.merge((ds_surf, ds_ml), compat='override')
    
    print('Read data from: era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.grib')
    print('Read data from: era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.grib')
    ds_merged.to_netcdf('era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'.nc')
    print('Written data to: ', 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'.nc')

    #os.remove('era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.grib')
    #os.remove('era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.grib')

'''
for r in zip(dates_for_each_time, times_for_each_day):
    print('started for the date:   ', r)
    merge_gribs(r[0], r[1])
'''
#Process grib files in parallel
if __name__ == '__main__':
    with Pool(8) as p:
        M = p.starmap(merge_gribs, zip(dates_for_each_time, times_for_each_day))
