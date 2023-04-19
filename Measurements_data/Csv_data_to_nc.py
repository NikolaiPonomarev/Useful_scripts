import csv
import numpy as np
import datetime
import pandas as pd
import math
import xarray as xr
import glob

#Path to the folder with csv data

DIR = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/'
# pattern = 'zuerich_sensor'

#Path to the meta file containing sensor ids and their category, e.g. mid_cost
sensor_file = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/helper_tables/sensors_table.csv'
ds_sensor = pd.read_csv(sensor_file).to_xarray()
mask_id = ds_sensor.sensor_group.values == 'mid_cost'
midcost_sensors_ids = ds_sensor.sensor_id.values[mask_id]
print(midcost_sensors_ids)
prefix = 'zuerich_sensor_network_sensor_time_series_'
suffix = '.csv'

# suffix = 'series_4??.csv'
# files = glob.glob(DIR + pattern + "*"+suffix)
files = np.array([DIR + prefix + str(idst) + suffix for idst in midcost_sensors_ids])

#Path to the meta file containing location names, coordinates, topography heights, etc.
meta_file = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/helper_tables/sites_table.csv'
ds_meta = pd.read_csv(meta_file).to_xarray()

filter_index = np.array(['sensor:field' in str(x) for x in ds_meta.site_purpose.values])

sensor_site_code_filter = ds_meta.site.values[filter_index]

# sites_file = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_data/mid_cost_sensor_observation_sites.csv'
# co2_file = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_data/mid_cost_co2_adj_cyl_sensor_observations.csv'

# #Create two datasets for site decription and CO2 data
# df_st = pd.read_csv(sites_file)
# df_co2 = pd.read_csv(co2_file)

# ds_st = df_st.to_xarray()
# ds_co2 = df_co2.to_xarray()

Output_dir = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/'

#Separate table with info about inlet heights above ground
sensor_heights_metafile = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/20230112_Sensor_Locations.xlsx'
ds_heights_meta = pd.read_excel(sensor_heights_metafile).to_xarray()

sensor_heights_metafile2 = '/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/processes_table.csv'
ds_heights_meta2 = pd.read_csv(sensor_heights_metafile2).to_xarray()

#Write data to a netcdf file for ecah station

# if np.unique(ds_co2.sensor_id.values).size == np.unique(ds_co2.site_name.values).size: #check that the number of unique sensor ids is equal to the number of unique sites
#     station_dim = np.arange((ds_st.site_name.values).size) 
# else:
#     print('Number of sites is not equal to the number of sensors')


exceptions_list = np.array(['429', '434'])
exceptions_codes = np.array(['due1', 'sott'])

variable_name_to_extract = 'co2_adj_cyl' #'co2_rep_adj_dry'


print(sensor_site_code_filter)

#Loop through csv files filtering out calibration data points and storing the result to the netcdf file
for f in files:

    df_co2 = pd.read_csv(f)
    ds_co2 = df_co2.to_xarray()
    print('DEBUG, Sensor id: ', ds_co2.sensor_id.values[:10])
    var_mask = ds_co2.variable.values== variable_name_to_extract
    # print(np.unique(ds_co2.site.values))   
    if str(ds_co2.sensor_id.values[1]) in exceptions_list:
        sitename_mask = np.unique(ds_co2.site.values)==exceptions_codes[np.where(exceptions_list == str(ds_co2.sensor_id.values[1]))]
    else:
        site_filter = np.array([x in sensor_site_code_filter for x in np.unique(ds_co2.site.values)])
        sitename_mask = np.logical_and(site_filter, np.unique(ds_co2.site.values) != 'unknown')
#     print(sitename_mask)
#     print(np.unique(ds_co2.site.values)[sitename_mask])    
    st_code = np.unique(ds_co2.site.values)[sitename_mask]
    # print(np.unique(ds_co2.site.values)[sitename_mask])
    for code in st_code:
        st_mask = ds_co2.site.values[var_mask] == code

        times_dim = np.arange(np.sum(st_mask))
        dates_str = (ds_co2.date.values[var_mask])[st_mask]
        dates_dt = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in dates_str])
        # define data with variable attributes
        data_vars = {'Concentration':(['time'], (ds_co2.value.values[var_mask])[st_mask], 
                                {'units': 'ppm', 
                                'long_name':'CO2_concentration'}),
                    'Dates':(['time'], dates_dt, 
                                { 
                                'long_name':'Dates'}),
                                
                                }
        # define coordinates
        coords = {'time': (['time'], times_dim )}
        #coords = {'station': (['station'], [1])}
        st_mask_attr = np.unique(ds_meta.site.values) == code
        sensor_id = (ds_co2.sensor_id.values[var_mask])[st_mask][0]

        # mask_names = np.where(ds_heights_meta.LocationName.values == code.upper())
        # mask_id = np.where(ds_heights_meta.LocationName.values == code.upper())
        
        h_ind = np.where(ds_heights_meta.LocationName.values == code.upper())[0]

        h_ind_midcost = h_ind[np.isin(ds_heights_meta.SensorUnit_ID.values[h_ind], midcost_sensors_ids)]
        # print(ds_heights_meta.LocationName.values[h_ind_midcost], code.upper(), h_ind_midcost)
        mask_meta_site = np.where(ds_heights_meta2.site.values == code)
        mask_var = np.where(ds_heights_meta2.variable.values[mask_meta_site] == variable_name_to_extract)
        mask_sensor = np.where((ds_heights_meta2.sensor_id.values[mask_meta_site])[mask_var] == sensor_id)
        inlet_h = ((ds_heights_meta2.height_above_ground_inlet.values[mask_meta_site])[mask_var])[mask_sensor]
        
        # inlet_h = ds_heights_meta['Inlet_HeightAbove Ground'].values[h_ind_midcost]
        real_srfc = ds_heights_meta.h.values[h_ind_midcost]
        # print(real_srfc, h_ind_midcost)
        if  code == 'due1':
            inlet_h = 0.
            real_srfc = real_srfc[0]
        else:
            print('DEBUG site code', code, h_ind_midcost, ds_heights_meta.h.values[h_ind_midcost])
            if h_ind_midcost.size>1:
                print(h_ind_midcost, real_srfc, real_srfc[0])
                if np.all(real_srfc == real_srfc[0]):
                    real_srfc = real_srfc[0]
            if inlet_h.size>1:
                if np.all(inlet_h == inlet_h[0]):
                    inlet_h = inlet_h[0] 
        masl_h = real_srfc + inlet_h

        print('INLET DEUBG', inlet_h, real_srfc, code)
        # print('DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(st_code, np.unique(ds_meta.site.values)[st_mask_attr], ds_meta.site.values[st_mask_attr])
        # print('DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # define global attributes
        print(ds_meta.site_name.values[st_mask_attr])
        attrs = {'Stations_name': ds_meta.site_name.values[st_mask_attr][0],      
                'Stations_id': ds_meta.site.values[st_mask_attr][0],
                'Sensor_id': sensor_id,  
                'Srfc_height': real_srfc,
                'Inlet_height_above_ground':inlet_h,
                'Lon':ds_meta.longitude.values[st_mask_attr][0],     
                'Lat':ds_meta.latitude.values[st_mask_attr][0], 
                'Inlet_height':masl_h,   
                'creation_date':str(datetime.datetime.now()), 
                'author':'Nikolai Ponomarev', 
                'email':'nikolai.ponomarev@empa.ch'}

        # create dataset
        ds_extracted_obs_matrix = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)

        name = Output_dir + 'ZHcyl_obs_' + ds_meta.site_name.values[st_mask_attr][0] + '_' + ((ds_co2.date.values[var_mask])[st_mask])[0][:-9] + '_' + ((ds_co2.date.values[var_mask])[st_mask])[-1][:-9] + '_' + str(sensor_id) + '.nc'
        print(name)
        #encoding = {
        #            'Concentration': {'_FillValue': np.nan,},
        #            'Std': {'_FillValue': np.nan,},
        #            'Dates': {'_FillValue': np.nan,}

        #           }


        ds_extracted_obs_matrix.to_netcdf(name)