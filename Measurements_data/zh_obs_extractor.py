import numpy as np
import glob
import xarray as xr
import datetime
from datetime import timedelta
import multiprocessing as mp
from multiprocessing import Pool

DIR = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/'
pattern = 'ZHcyl_obs_'
files = glob.glob(DIR + pattern + "*")

number_of_stations = len(files)

print('Aquiring data from ' + str(number_of_stations) + ' files: ', files)

startdate = datetime.datetime(2022, 7, 1, 0)
enddate = datetime.datetime(2022, 12, 23, 23)
delta = enddate - startdate
#factor_co2 = 28.97/44.
factor_co2 = 1.
toppm_dict = {'nmol mol-1':factor_co2*1e-9*1e6, 
               'umol mol-1':factor_co2, 
               'µmol mol-1':factor_co2, 
             }

chosen_dates = []
count = 0
for i in range(delta.days + 1):
    
    day = startdate + timedelta(days=i)
    for h in range(0,24):
        count = count+1
        day1 = day + timedelta(hours=h)
        chosen_dates.append(np.datetime64(day1.strftime('%Y-%m-%dT%H:%M:%S.000000000')))
        print(day1, count)

number_of_hourly_measurements = count #- 23 # -23 to not account for the last 23 hours of the end date



obs_cnc_matrix = np.zeros((number_of_stations, number_of_hourly_measurements), dtype=np.float64)
obs_dates_matrix = np.zeros((number_of_stations, number_of_hourly_measurements), dtype = np.dtype('datetime64[ns]'))

def extract_obs_column(file):
    print('Opened file ', file)
    obs_cnc1= np.empty((number_of_hourly_measurements), dtype=np.float64)
    obs_cnc1[:] = np.nan
    obs_dates1 = np.empty((number_of_hourly_measurements), dtype = np.dtype('datetime64[ns]'))
    obs_dates1[:] = np.datetime64("NaT")
    obs_std1 = np.empty((number_of_hourly_measurements), dtype=np.float64)
    obs_std1[:] = np.nan
  
    ds = xr.open_dataset(file)
     
    lons = ds.attrs['Lon']
    lats = ds.attrs['Lat'] 
    name = ds.attrs['Stations_name']
    id_st = ds.attrs['Sensor_id']
    units = 'ppm_dry'
    masl = ds.attrs['Inlet_height']
    inlet_abg = ds.attrs['Inlet_height_above_ground']
    srfc_h = ds.attrs['Srfc_height']
    #print(masl)
    diff=(ds.Dates.values[1]-ds.Dates.values[0])/3600000000000
    if  diff!=1:
        print('Observation data at station',name, 'is not hourly averaged', diff)
    #cnc[ cnc < 0 ] = np.nan
    start = np.where(ds.Dates.values >= np.datetime64(startdate.strftime('%Y-%m-%dT%H:%M:%S.000000000')))[0]
    end = np.where(ds.Dates.values <= np.datetime64(enddate.strftime('%Y-%m-%dT%H:%M:%S.000000000')))[0]
    if len(start)>0 and len(end)>0:
        ind_start = start[0]
        # print('START', start)
        #starting_time = ((ds.TIMESTAMP.values[ind_start] - np.datetime64(startdate.strftime('%Y-%m-%dT%H:%M:%S.000000000')))/3600000000000).astype(int)
        #print(ds.Dates.values, name, np.datetime64(startdate.strftime('%Y-%m-%dT%H:%M:%S.000000000')))
        ind_end = end[-1]
        #ending_time = number_of_hourly_measurements - ((np.datetime64(enddate.strftime('%Y-%m-%dT%H:%M:%S.000000000')) - ds.TIMESTAMP.values[ind_end])/3600000000000).astype(int)
        #print(ind_start, ind_end, ind_start - ind_end, ((np.datetime64(enddate.strftime('%Y-%m-%dT%H:%M:%S.000000000')) - ds.TIMESTAMP.values[ind_end])/3600000000000).astype(int), starting_time, ending_time, ds.TIMESTAMP.values[ind_start], ds.TIMESTAMP.values[ind_end])
        #print(ind_start, ind_end, ind_end-ind_start, ds.TIMESTAMP.values[ind_start], ds.TIMESTAMP.values[ind_end], number_of_hourly_measurements)
        for i in range(ind_start, ind_end+1):
            try:
                index = np.where(chosen_dates==ds.Dates.values[i])[0][0]
                obs_dates1[index] = ds.Dates.values[i]
                #print(Dates.values[i])
                #obs_std1[index] = ds.Stdev.values[i]#*toppm_dict[units]     - data is already in ppm dry
                obs_cnc1[index] = ds.Concentration.values[i]#*toppm_dict[units]
            except:
                continue


    
    
    return name, obs_cnc1, obs_dates1, lons, lats, masl, id_st, inlet_abg, srfc_h 



if __name__ == '__main__':
    args =  files
    with Pool(36) as pool:
        M = list(zip(*pool.map(extract_obs_column, args))) 


station_names = [x for x in M[0]]
station_ids = [x for x in M[6]]
obs_cnc = [x for x in M[1]] 
obs_times = [x for x in M[2]]
obs_lons = [x for x in M[3]]
obs_lats = [x for x in M[4]]
obs_masl = [x for x in M[5]]
inlet_h = [x for x in M[7]]
srfc_h = [x for x in M[8]]

mask_true = np.repeat(True, len(obs_cnc_matrix[0]))
stations_to_be_removed = []
print('LON LAT for observations: ', obs_lons, obs_lats)
#check and remove stations without observations
for ix, x in enumerate(obs_cnc):
    if any(np.isfinite(x)) and (obs_lons[ix] < 8.935) and (obs_lons[ix] > 8.065) and (obs_lats[ix] < 47.605) and (obs_lats[ix] > 47.195):
        np.place(obs_cnc_matrix[ix], mask_true, x)
        np.place(obs_dates_matrix[ix], mask_true, obs_times[ix])
    else:    
        stations_to_be_removed.append(ix)
        print('DEBUG Removed station data ', x, obs_lons[ix], obs_lats[ix])

print('Sensor ids of removed stations: ', np.array(station_ids)[stations_to_be_removed], any(np.isfinite(x)))
obs_cnc_matrix = np.delete(obs_cnc_matrix, stations_to_be_removed, axis=0)
station_names = np.delete(station_names, stations_to_be_removed)
station_ids = np.delete(station_ids, stations_to_be_removed)
inlet_h = np.delete(inlet_h, stations_to_be_removed)
srfc_h = np.delete(srfc_h, stations_to_be_removed)
print(np.shape(obs_dates_matrix), obs_dates_matrix, obs_cnc_matrix)
obs_dates_matrix = np.delete(obs_dates_matrix, stations_to_be_removed, axis=0)
print(np.shape(obs_dates_matrix))
obs_lons = np.delete(obs_lons, stations_to_be_removed)
obs_lats = np.delete(obs_lats, stations_to_be_removed)
obs_masl = np.delete(obs_masl, stations_to_be_removed)
print('Dates shape:', np.shape(obs_times), obs_masl)
station_idcs = np.arange(len(station_ids))
# define data with variable attributes
data_vars = {'Concentration':(['station', 'time'], obs_cnc_matrix, 
                         {'units': 'ppm', 
                          'long_name':'CO2_concentration'}),
             'Stations_names':(['station'], station_names, 
                         {'units': '-', 
                          'long_name':'Stations_names'}),     
             'Sensor_id':(['station'], station_ids, 
                         {'units': '-', 
                          'long_name':'Sensor_ids'}),
             'Stations_masl':(['station'], obs_masl, 
                         {'units': '-', 
                          'long_name':'Elevation_heights_above_sl'}),     
             'Lon':(['station'], obs_lons, 
                         {'units': 'degrees', 
                          'long_name':'Longituted'}),     
             'Lat':(['station'], obs_lats, 
                         {'units': 'degrees', 
                          'long_name':'Latituted'}),
             'Dates':(['station', 'time'], obs_dates_matrix, 
                         { 
                          'long_name':'Dates'}),
             'Inlet_h_above_ground':(['station'], inlet_h, 
                         { 
                          'long_name':'Inlet_height_above_ground'}),     
             'Srfc_h':(['station'], srfc_h, 
                         { 
                          'long_name':'Surface_height'}),        
                          }

# define coordinates
coords = {'time': (['time'], obs_times)}
coords = {'station': (['station'], station_idcs)}

# define global attributes
attrs = {'creation_date':str(datetime.datetime.now()), 
         'author':'Nikolai Ponomarev', 
         'email':'nikolai.ponomarev@empa.ch'}

# create dataset
ds_extracted_obs_matrix = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

name = 'Extracted_' + pattern + '_' + startdate.strftime('%Y%m%d') + '_' + enddate.strftime('%Y%m%d') + 'alldates_masl_inlet_cyl.nc'

#encoding = {
#            'Concentration': {'_FillValue': np.nan,},
#            'Std': {'_FillValue': np.nan,},
#            'Dates': {'_FillValue': np.nan,}

 #           }


ds_extracted_obs_matrix.to_netcdf(name)

print('Finished extraction and stored obs_matrix for : '+str(len(obs_lons)) + ' stations (from' + str(number_of_stations) +' available ICOS stations), which were operating during the given period and are located inside of the model damain, in the file: ', name)
