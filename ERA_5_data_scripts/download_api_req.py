import cdsapi
import datetime
import numpy as np
from multiprocessing import Pool
from multiprocessing import Pool
import xarray as xr
import  os, sys
import glob

#Generate dates for which you want to get the data
start_day = datetime.datetime.strptime("2023-01-01", "%Y-%m-%d")
end_day = datetime.datetime.strptime("2023-03-27", "%Y-%m-%d")

generated_dates = [start_day + datetime.timedelta(days=x) for x in range(0, (end_day-start_day).days+1)]

#Generate times of the day for which you want to get the data
start = datetime.datetime.strptime("2022-01-01:00:00", "%Y-%m-%d:%H:%M")
end = datetime.datetime.strptime("2022-01-01:23:00", "%Y-%m-%d:%H:%M")
generated_times = np.array([datetime.time(i,0) for i in range(start.hour,end.hour+1)])

print('Given dates ', generated_dates, generated_times)
pattern = 'era5_'
#Get the list of era5 files that are already present in the directory
files = glob.glob(pattern + "*")

def fetch_era5(date, time, dir2move='/scratch/snx3000/nponomar/ERA5'):

    #Check if one of the files for the given date / time of the day already exists
    if ('era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.grib' not in files) or ('era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.grib' not in files):
        """Fetch ERA5 data from ECMWF for initial conditions

        Parameters
        ----------
        date : initial date to fetch

        """

        c = cdsapi.Client()

        # -- CRWC : Specific rain water content              - 75
        # -- CSWC : Specific snow water content              - 76
        # -- T    : Temperature                             - 130
        # -- U    : U component of wind                     - 131
        # -- V    : V component of wind                     - 132
        # -- Q    : Specific humidity                       - 133
        # -- W    : Vertical velocity                       - 135
        # -- CLWC : Specific cloud liquid water content     - 246
        # -- CIWC : Specific cloud ice water content        - 247

        c.retrieve('reanalysis-era5-complete', {
            'class': 'ea',
            'date': date.strftime('%Y-%m-%d'),
            'time': time.strftime('%H:%M:%S'),
            'expver': '1',
            'levelist': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137',
            'levtype': 'ml',
            'param': '75/76/129/130/131/132/133/135/246/247',
            'stream': 'oper',
            'type': 'an',
            'area' : [
                60, -15, 35, 
                    20],
            'grid': '0.25/0.25',             
        }, 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_ml.grib')

        # -- CI   : Sea Ice Cover                   - 31
        # -- ASN  : Snow albedo                     - 32
        # -- RSN  : Snow density                    - 33
        # -- SST  : Sea Surface Temperature         - 34
        # -- SWV1 : Volumetric soil water layer 1   - 39
        # -- SWV2 : Volumetric soil water layer 2   - 40
        # -- SWV3 : Volumetric soil water layer 3   - 41 
        # -- SWV4 : Volumetric soil water layer 4   - 42
        # -- Z    : Geopotential                   - 129
        # -- SP   : Surface pressure               - 134 --> Must be converted to logarithm
        # -- STL1 : Soil temperature level 1       - 139 
        # -- SD   : Snow depth                     - 141
        # -- STL2 : Soil temperature level 2       - 170
        # -- LSM  : Land-Sea Mask                  - 172
        # -- STL3 : Soil temperature level 3       - 183 
        # -- SRC  : Skin reservoir content         - 198
        # -- SKT  : Skin Temperature               - 235
        # -- STL4 : Soil temperature level 4       - 236 
        # -- TSN  : Temperature of snow layer      - 238
        #--  SLT  : Soil type                      - 43

        c.retrieve('reanalysis-era5-complete', {
            'class': 'ea',
            'date': date.strftime('%Y-%m-%d'),
            'time': time.strftime('%H:%M:%S'),
            'expver': '1',
            'levtype': 'sfc',
            'param': '43/31.128/32.128/33.128/34.128/39.128/40.128/41.128/42.128/129.128/134.128/139.128/141.128/170.128/172.128/183.128/198.128/235.128/236.128/238.128',
            'stream': 'oper',
            'type': 'an',
            'area' : [
                60, -15, 35, 
                    20],
            'grid': '0.25/0.25',
        }, 'era5_'+date.strftime('%Y%m%d')+"{:02d}".format(time.hour)+'_surf.grib')
        print('Downloaded data for the date: ', date.strftime('%Y%m%d'), "{:02d}".format(time.hour))
    else:
        print(date, ' file already exists')

#for d in generated_dates:
#    for t in generated_times:

times_for_each_day = np.repeat([np.array(generated_times)], len(generated_dates), axis=0).flatten()
dates_for_each_time = np.repeat(generated_dates, len(generated_times))

#Request data in parallel for different days / times of the day
if __name__ == '__main__':
    with Pool(42) as p:
        M = p.starmap(fetch_era5, zip(dates_for_each_time, times_for_each_day))
        #fetch_era5(d, t, '/scratch/snx3000/nponomar/ERA5')
        
