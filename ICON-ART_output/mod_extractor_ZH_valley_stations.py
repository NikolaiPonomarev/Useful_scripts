import numpy as np
import glob
import xarray as xr
import datetime
from datetime import timedelta
import multiprocessing as mp
from multiprocessing import Pool
import math
from itertools import repeat
PATH = '/scratch/snx3000/nponomar/processing_chain_python/processing-chain/work/VPRM_ZH_22/2022070100_0_4248/icon/output/'
# PATH = '/scratch/snx3000/nponomar/processing_chain_python/processing-chain/work/VPRM_EU_ERA5_22/2022070100_0_4248/icon/output/'
pattern = 'ICON-ART-UNSTRUCTURED_DOM01_'
# pattern = 'ICON-ART-UNSTR_DOM01_'
#pattern2 = 'ICON-ART-OEM_DOM01_'

# ds_obs = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_ZH_obs__20220701_20220731alldates_masl.nc')
ds_obs = xr.open_dataset('/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/Extracted_ZHcyl_obs__20220701_20221223alldates_masl_inlet_cyl.nc')
# ds_obs = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_METEO_ZH_obs_METEO__20220701_20221223ZH_windsensors.nc')
obs_asl = ds_obs.Stations_masl.values
real_srfc = ds_obs.Srfc_h.values
obs_inlet = ds_obs.Inlet_h_above_ground.values
obs_lon = ds_obs.Lon.values
obs_lat = ds_obs.Lat.values
obs_st = ds_obs.Stations_names.values
startdate = datetime.datetime(2022, 7, 1, 0)
enddate = datetime.datetime(2022, 12, 23, 23)
delta = enddate - startdate
chosen_dates = []
for i in range(delta.days + 1):
    
    day = startdate + timedelta(days=i)
    for h in range(0,24):
        day1 = day + timedelta(hours=h)
        chosen_dates.append(day1.strftime('%Y%m%dT%H'))
chosen_dates = np.asarray(chosen_dates)
ds_grid = xr.open_dataset('/scratch/snx3000/nponomar/processing_chain_python/processing-chain/work/VPRM_ZH_22/2022070100_0_744/icon/input/grid/icon_Zurich_R19B9_wide_DOM01.nc')
# ds_grid = xr.open_dataset('/users/nponomar/icon-art/icon/grids/icon_europe_DOM01.nc')
len_factor_lat = 110.574
len_factor_lon = 111.320
lats_rad = ds_grid['clat'].values
lons_rad = ds_grid['clon'].values
print('Reading input grid, number of lats and lons:', len(lats_rad), len(lons_rad))

lons_deg = np.rad2deg( lons_rad )
lats_deg = np.rad2deg( lats_rad )

obs_lat_rad = np.deg2rad( obs_lat )

lons_km = np.asarray([x*math.cos(lats_rad[ix])*len_factor_lon for ix, x in enumerate(lons_deg)])
lats_km = np.asarray([x*len_factor_lat for x in lats_deg])
obs_lon_km = np.asarray([x*math.cos(obs_lat_rad[ix])*len_factor_lon for ix, x in enumerate(obs_lon)])
obs_lat_km = np.asarray([x*len_factor_lat for x in obs_lat])

def extract_mod_first_iter():
    
    file = PATH + pattern + chosen_dates[0] + '0000Z.nc'

    print('Reading file with model ouput: ', file)
    ds_mod = xr.open_dataset(file)
    number_of_heights = 5
    number_of_cells = 5
    factor = 1e6*28.97/44.
    z_asl = ds_mod.z_mc.values #+ ds_mod.topography_c.values 
    #Calculate the total CO2 concentration
    #cnc_mod = (ds_mod.TRCO2_Anthropogenic_chemtr[0, :] + ds_mod.TRCO2_BG_chemtr[0, :] +  ds_mod.CO2_RA[0, :] - ds_mod.CO2_GPP[0, :]).values*factor
    # cnc_mod = [ds_mod.TRCO2_A_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG[0, :].values*factor, ds_mod.CO2_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), - ds_mod.CO2_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values)]
    # cnc_mod = [ds_mod.TRCO2_Anthropogenic_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.CO2_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), - ds_mod.CO2_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values)]
    cnc_mod = [ds_mod.TRCO2_A_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG[0, :].values*factor/(1.-ds_mod.qv[0, :].values),
                ds_mod.CO2_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), - ds_mod.CO2_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values),
                ds_mod.TRCO2_BG_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values)]
    cnc_interp = np.zeros((len(cnc_mod), len(obs_st)), dtype=np.float64)
    std_interp = np.zeros((len(cnc_mod), len(obs_st)), dtype=np.float64)
    shape_list = (len(cnc_mod), len(obs_st))

    ind_hor = np.empty(shape=shape_list+(0,)).tolist()
    ind_vert = np.empty(shape=shape_list+(0,)).tolist()
    dist_hor = np.empty(shape=shape_list+(0,)).tolist()
    dist_vert = np.empty(shape=shape_list+(0,)).tolist()
    mod_s_h = []
    z_ifc = ds_mod.z_ifc.values[1:, :]
    z_ifc_top = ds_mod.z_ifc.values[:-1, :]
    for station in np.arange(len(obs_st)):
            cnc_interp[0][station], std_interp[0][station], ind_hr, ind_vrt, dist_vrt, dist_hr, msfch = interp_mod_first_iteration(station, cnc_mod[0], z_asl, number_of_heights, number_of_cells, ds_mod, z_ifc, z_ifc_top)
            ind_hor[0][station].extend(ind_hr)
            ind_vert[0][station].extend(ind_vrt)
            dist_hor[0][station].extend(dist_hr)
            dist_vert[0][station].extend(dist_vrt)
            mod_s_h.append(msfch)

            # print(obs_st[station], ' inlet located at ', obs_inlet[station], ' m above ground', '\n', 'chosen model levels: ', ind_vrt, '\n', 'Method : ', m, '\n', 'chosen model heights above ground: ', hag, '\n', 'DWI coefficients: ', coeffs, '\n')

    return  cnc_interp, std_interp, ind_hor, ind_vert, dist_hor, dist_vert, mod_s_h


def interp_mod_first_iteration(station, cnc_mod_sp, z_asl, number_of_heights, number_of_cells, ds_mod, z_ifc, z_ifc_top):
    h_jung = 3000
    mod_lon = lons_km
    mod_lat = lats_km
           
    distances = np.sqrt((mod_lon - obs_lon_km[station])*(mod_lon - obs_lon_km[station]) + (mod_lat - obs_lat_km[station])*(mod_lat - obs_lat_km[station]))
    zero_dist_ind = np.where(distances==0)[0]
    cnc_interp = np.zeros((len(cnc_mod_sp)), dtype=np.float64)
    std_interp = np.zeros((len(cnc_mod_sp)), dtype=np.float64)
    z_interp = np.zeros((len(cnc_mod_sp)), dtype=np.float64)
    z_ifc_interp = np.zeros((len(cnc_mod_sp)), dtype=np.float64)
    z_ifc_top_interp = np.zeros((len(cnc_mod_sp)), dtype=np.float64)
    #get lowest boundary of model level heights

    for z_coord in range (len(cnc_mod_sp)):
        if np.size(zero_dist_ind)>0:
            cnc_interp[z_coord] = cnc_zinterp[z_coord, zero_dist_ind[0]]
            std_interp[z_coord] = 0 
            indx_h = zero_dist_ind[0]  
        else:
            indx = np.argsort(distances)
            indx_h = indx[0:number_of_cells]
            if np.isnan(cnc_mod_sp[z_coord, indx_h]).any(): #check for if the station is on the edge of the domain and there are boundary cells with nan values
                conc_station= cnc_mod_sp[z_coord, indx_h]
                dist_station = distances[indx_h]
                zasl_m = z_asl[z_coord, indx_h]
                z_ifc_top_m = z_ifc_top[z_coord, indx_h]
                z_ifc_m = z_ifc[z_coord, indx_h]
                mask_ind = np.logical_not(np.ma.masked_invalid(conc_station).mask)
                indx_h = indx_h[mask_ind]
                u = np.nansum(conc_station[mask_ind]/dist_station[mask_ind]) #summ by i of a product: cnc_i and 1/distance_i
                uz = np.nansum(zasl_m[mask_ind]/dist_station[mask_ind])
                uz_ifc = np.nansum(z_ifc_m[mask_ind]/dist_station[mask_ind])
                uz_ifc_top = np.nansum(z_ifc_top_m[mask_ind]/dist_station[mask_ind])
                w = np.nansum(1.0/dist_station[mask_ind]) #summ by i of weights - 1/distance_i
                cnc_interp[z_coord] = u/w
                z_interp[z_coord] = uz/w
                z_ifc_interp[z_coord] = uz_ifc/w
                z_ifc_top_interp[z_coord] = uz_ifc_top/w
                std_interp[z_coord] = np.std(conc_station[mask_ind])
            else:    
                u = np.nansum(cnc_mod_sp[z_coord, indx_h]/distances[indx_h]) #summ by i of a product: cnc_i and 1/distance_i
                w = np.nansum(1.0/distances[indx_h]) #summ by i of weights - 1/distance_i
                uz_ifc = np.nansum(z_ifc[z_coord, indx_h]/distances[indx_h])
                uz_ifc_top = np.nansum(z_ifc_top[z_coord, indx_h]/distances[indx_h])
                uz = np.nansum(z_asl[z_coord, indx_h]/distances[indx_h])
                z_interp[z_coord] = uz/w
                z_ifc_interp[z_coord] = uz_ifc/w
                z_ifc_top_interp[z_coord] = uz_ifc_top/w
                cnc_interp[z_coord] = u/w
                std_interp[z_coord] = np.std(cnc_mod_sp[z_coord, indx_h])


    #Vertical interpolation
    # if obs_st[station] == 'Jungfraujoch':
    #     z_dist = abs(z_interp - h_jung)  #Patch for Jungfraujoch special treatment
    
    # else:        
    z_dist = abs(z_interp - obs_asl[station])

    z = z_interp - z_ifc_interp[-1]
    # z1 = z_ifc_top_interp - z_ifc_interp
    zero_zdist_ind = np.where(z_dist==0)[0]
    if real_srfc[station] < z_ifc_interp[-1]: #obs_asl[station] < z_ifc_interp[-1]
        
        # print('special case, valley station-', obs_st[station])
        if obs_inlet[station] < z[-1]:
            # method = 'new method for valley stations, inlet height ', obs_inlet[station], ' is lower than the middle of the first layer ', z[-1]
            cnc_zinterp = cnc_interp[-1]
            std_zinterp = 0
            indx_z = [-1]
            
        else:
            # if obs_inlet[station]>35:
            #     inlet = obs_inlet[station]/2
            # else:
            # method = 'new method for valley stations, inlet height ', obs_inlet[station], ' is higher than the middle of the first layer ', z[-1]
            inlet = obs_inlet[station]
            sampling_dist = abs(z - inlet)
            closest_height_ind = np.argsort(sampling_dist)
            indx_z = closest_height_ind[0:number_of_heights]
            # print('DEBUG!!!!', z[indx_z], inlet,  indx_z)
            # closest_height_ind = np.nanargmin(sampling_dist)
            # cnc_zinterp = cnc_interp[closest_height_ind]
            uz = np.nansum(cnc_interp[indx_z]/sampling_dist[indx_z]) #summ by i of a product: cnc_i and 1/distance_i
            wz = np.nansum(1.0/sampling_dist[indx_z]) #summ by i of weights - 1/distance_i
            cnc_zinterp = uz/wz
            std_zinterp = math.sqrt((np.std(cnc_interp[indx_z]))**2+(np.sum((std_interp[indx_z])**2)/number_of_heights)**2)
            # k = 1/np.array(sampling_dist[indx_z]*wz)
            # std_zinterp = 0
            # indx_z = [closest_height_ind]
    
    else:    
        if np.size(zero_zdist_ind)>0:
                # method = 'old method ', obs_inlet[station], ' sampling height is equal to the model level height ', z_interp[zero_zdist_ind[0]]
                cnc_zinterp = cnc_interp[zero_zdist_ind[0]]
                std_zinterp = std_interp[zero_zdist_ind[0]]
                indx_z = zero_zdist_ind[0]
                
        else:      
                
                indx_z = np.argsort(z_dist)
                indx_z = indx_z[0:number_of_heights]
                # method = 'old DWI method, using masl heights for interpolation, station masl = ', obs_asl[station], 'chosen levels masl: ', z_interp[indx_z]
                uz = np.nansum(cnc_interp[indx_z]/z_dist[indx_z]) #summ by i of a product: cnc_i and 1/distance_i
                wz = np.nansum(1.0/z_dist[indx_z]) #summ by i of weights - 1/distance_i
                cnc_zinterp = uz/wz
                
                std_zinterp = math.sqrt((np.std(cnc_interp[indx_z]))**2+(np.sum((std_interp[indx_z])**2)/number_of_heights)**2)
    #print(cnc_zinterp)    
    #Horisontal interpolation

    #print('Finished interpolation')
    return cnc_zinterp, std_zinterp, indx_h, indx_z, z_dist[indx_z], distances[indx_h], z_ifc_interp[-1]

mod_cnc_full = []
mod_std_full = []

mod_cnc_, mod_std_, indicies_horizontal, indicies_vertical, distances_horizontal, distances_vertical, model_surface_height = extract_mod_first_iter()

# print(model_surface_height, np.shape(model_surface_height))

#mod_cnc_full.append(mod_cnc_)
#mod_std_full.append(mod_std_)

def extract_mod_predetermined_indcs(args):
    
    file = PATH + pattern + args + '0000Z.nc'
    h_ind = indicies_horizontal 
    v_ind = indicies_vertical 
    h_dist = distances_horizontal
    v_dist = distances_vertical
    print('Reading file with model ouput: ', file)
    ds_mod = xr.open_dataset(file)
    factor = 1e6*28.97/44.
    z_asl = ds_mod.z_mc.values #+ ds_mod.topography_c.values
    #Calculate the total CO2 concentration
    #cnc_mod = (ds_mod.TRCO2_Anthropogenic_chemtr[0, :] + ds_mod.TRCO2_BG_chemtr[0, :] +  ds_mod.CO2_RA[0, :] - ds_mod.CO2_GPP[0, :]).values*factor
    # cnc_mod = [ds_mod.TRCO2_A_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG[0, :].values*factor, ds_mod.CO2_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), - ds_mod.CO2_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values)]
    cnc_mod = [ds_mod.TRCO2_A_chemtr[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG[0, :].values*factor/(1.-ds_mod.qv[0, :].values),
                ds_mod.CO2_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), - ds_mod.CO2_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values),
                ds_mod.TRCO2_BG_RA[0, :].values*factor/(1.-ds_mod.qv[0, :].values), ds_mod.TRCO2_BG_GPP[0, :].values*factor/(1.-ds_mod.qv[0, :].values)]
    cnc_interp = np.zeros((len(cnc_mod), len(obs_st)), dtype=np.float64)
    std_interp = np.zeros((len(cnc_mod), len(obs_st)), dtype=np.float64)
    for ispecie, specie in enumerate(cnc_mod):
        for station in np.arange(len(obs_st)):
            cnc_interp[ispecie][station], std_interp[ispecie][station] = int_predetermined_indcs(specie, h_ind[0][station], v_ind[0][station], h_dist[0][station], v_dist[0][station])


    return  cnc_interp, std_interp


def int_predetermined_indcs(c, h, z, h_d, z_d):
    
    distances = np.array(h_d)
    cnc_interp = np.zeros((len(c)), dtype=np.float64)
    std_interp = np.zeros((len(c)), dtype=np.float64)
    for z_coord in range (len(c)):
        if np.size(h)<2:
            cnc_interp[z_coord] = c[z_coord, h]
            std_interp[z_coord] = 0  
        else:

            if np.isnan(c[z_coord, h]).any(): #check for if the station is on the edge of the domain and there are boundary cells with nan values
                conc_station= c[z_coord, h]
                dist_station = distances
                mask_ind = np.logical_not(np.ma.masked_invalid(conc_station).mask)
                u = np.nansum(conc_station[mask_ind]/dist_station[mask_ind]) #summ by i of a product: cnc_i and 1/distance_i
                w = np.nansum(1.0/dist_station[mask_ind]) #summ by i of weights - 1/distance_i
                cnc_interp[z_coord] = u/w
                std_interp[z_coord] = np.std(conc_station[mask_ind])
            else:    
                u = np.nansum(c[z_coord, h]/distances) #summ by i of a product: cnc_i and 1/distance_i
                w = np.nansum(1.0/distances) #summ by i of weights - 1/distance_i
                cnc_interp[z_coord] = u/w
                std_interp[z_coord] = np.std(c[z_coord, h])

    #Vertical interpolation    

    z_dist = np.array(z_d)

    if np.size(z)<2:
            cnc_zinterp = cnc_interp[z]
            std_zinterp = std_interp[z]
    else:       
            indx_z = z
            uz = np.nansum(cnc_interp[indx_z]/z_dist) #summ by i of a product: cnc_i and 1/distance_i
            wz = np.nansum(1.0/z_dist) #summ by i of weights - 1/distance_i
            cnc_zinterp = uz/wz
            std_zinterp = math.sqrt((np.std(cnc_interp[indx_z]))**2+(np.sum((std_interp[indx_z])**2)/len(z))**2)
    return cnc_zinterp, std_zinterp



if __name__ == '__main__':
    args =  chosen_dates
    with Pool(36) as pool:
        M = list(zip(*pool.map(extract_mod_predetermined_indcs, args))) 


mod_cnc = [x for x in M[0]] 
mod_std =[x for x in M[1]]
mod_cnc_full.extend(mod_cnc)
mod_std_full.extend(mod_std)

print(np.shape(np.transpose(mod_cnc_full)), mod_cnc_full)


station_idcs = np.arange(len(obs_st))
specie_idcs = np.arange(len(np.transpose(mod_cnc)[0, :, 0]))

# define data with variable attributes
data_vars = {'Concentration':(['station', 'specie', 'time'], np.transpose(mod_cnc), 
                         {'units': 'ppm', 
                          'long_name':'CO2_concentration'}),
             'Std':(['station', 'specie', 'time'], np.transpose(mod_std), 
                         {'units': 'ppm', 
                          'long_name':'CO2_intepolation_std'}),  
             'Stations_names':(['station'], obs_st, 
                         {'units': '-', 
                          'long_name':'Stations_names'}),         
             'Lon':(['station'], obs_lon, 
                         {'units': 'degrees', 
                          'long_name':'Longituted'}),     
             'Lat':(['station'], obs_lat, 
                         {'units': 'degrees', 
                          'long_name':'Latituted'}),
             'Srfc_model_height':(['station'], model_surface_height, 
                         {'units': 'm', 
                          'long_name':'Surface_height_in_model'}),
                          
                          }

# define coordinates
coords = {'time': (['time'], chosen_dates)}
coords = {'station': (['station'], station_idcs)}
coords = {'specie': (['specie'], specie_idcs)}
# define global attributes
attrs = {'creation_date':str(datetime.datetime.now()), 
         'Model_surfc_height_in_masl':'Nikolai Ponomarev',
         'author':'Nikolai Ponomarev', 
         'email':'nikolai.ponomarev@empa.ch'}

# create dataset
ds_extracted_obs_matrix = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

name = 'Extracted_DWI_' + pattern + '_' + chosen_dates[0] + '_' + chosen_dates[-1] + '_ZH_Tcoeffs.nc'

encoding = {
            'Concentration': {'_FillValue': np.nan,},
            'Std': {'_FillValue': np.nan,}

            }

print('Writing output to the  : ', name)
ds_extracted_obs_matrix.to_netcdf(name, encoding = encoding)

print('Finished extraction and stored mod_matrix in the file: ', name)
