import itertools
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np
import os
from cbar_discrete import custom_cb
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.patches as mpatches
import datetime
from datetime import timedelta
import pandas as pd
import calendar
import math 
from itertools import repeat
from dateutil.relativedelta import relativedelta

MYDIR = ("/scratch/snx3000/nponomar/plt_py/ICONvsOBS_ZH2022/WeeklyBias_EU_old_int")
CHECK_FOLDER = os.path.isdir(MYDIR)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
    print("created folder : ", MYDIR)

# ds_obs = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_ZHcyl_obs__20220701_20221223alldates_masl_inlet_cyl.nc')
ds_obs = xr.open_dataset('/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/Extracted_ZHcyl_obs__20220701_20221223alldates_masl_inlet_cyl.nc')
# ds_mod1 = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_DWI_ICON-ART-UNSTRUCTURED_DOM01__20220701T00_20221223T23_ZH_Corine_LBC.nc')
# ds_mod1 = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_DWI_ICON-ART-UNSTRUCTURED_DOM01__20220701T00_20221223T23_ZH_Damping.nc')
# ds_mod2 = xr.open_dataset('/scratch/snx3000/nponomar/ICOS_obs_data/ZH_obs_gdrive/decompressed_data/Extracted_DWI_ICON-ART-UNSTRUCTURED_DOM01__20220701T00_20221223T23_ZH_valley_stations.nc')
# label_mod1 = 'ICON-ART-VPRM, EU, no qv'
ds_mod1 = xr.open_dataset('/scratch/snx3000/nponomar/plt_py/Extracted_DWI_ICON-ART-UNSTRUCTURED_DOM01__20220701T00_20221223T23_EU_wetdrylbc_forZH.nc')
cnc_mod1 = ds_mod1.Concentration.values #-23

cnc_obs = ds_obs.Concentration.values

lim_0 = 12
lim_1 = 15

suffix = 'Daily' +str(lim_0) + '_' + str(lim_1) + 'PM'

print(np.shape(cnc_mod1), np.shape(cnc_obs))

# averaging_step = 7 * 24 #one week for hourly data

# dates_index = np.arrange(ds_mod1.time.size)

stations_obs = ds_obs.Stations_names.values

stindex = np.arange(len(stations_obs))
Sensor_ids = ds_obs.Sensor_id.values

def weekly_avrg(st):

    dates = pd.to_datetime(ds_obs.Dates.values[st])   

    # 0 # Anthropogenic contribution
    # 1 # IC+BC contribution 
    # 2 # RA contribution
    # 3 # GPP contribution, negative vals

    st_name = stations_obs[st]
    sensorid = Sensor_ids[st]

    #print(np.shape(cnc_mod1[st, :, :]), np.shape(np.sum(cnc_mod1[st, :, :], axis = 0)), np.shape(np.sum(cnc_mod1[st, :, :], axis = 1)))

    cnc_o = cnc_obs[st]
    cnc_m = np.sum(cnc_mod1[st, :, :], axis = 0)
    
    bias =  cnc_m - cnc_o

    mask_obs = np.isfinite(bias)

    bias_no_gaps = bias[mask_obs]
    dates_no_gaps = dates[mask_obs]

    mask_time_of_the_day = [lim_0<=x.hour<=lim_1 for x in dates_no_gaps]
   

    # tstmp = np.array([pd.to_datetime(x) for x in dates[mask_obs]])
    tstmp = dates_no_gaps[mask_time_of_the_day]
    bias_no_gaps_times = bias_no_gaps[mask_time_of_the_day]
    
    week_number = np.array([x.isocalendar()[1] for x in tstmp])

    weekly_bias = np.array([np.nanmean(bias_no_gaps_times[np.where(week_number == x)]) for x in range(np.min(week_number), np.max(week_number) + 1)])

    #
    # if st_name == 'Albisgüetli':
    #     print('DEBUG!!!!!!!!', st_name, week_number, weekly_bias, (cnc_m[mask_obs])[mask_time_of_the_day], (cnc_o[mask_obs])[mask_time_of_the_day])

    return weekly_bias, week_number, st_name, sensorid


if __name__ == '__main__':
    args = stindex
    with Pool(8) as pool:
        M = list((pool.map(weekly_avrg, args)))


M = np.array(M)

weekly_bias_stations = np.array([x for x in M[:, 0]])

weekly_mean = np.array([np.nanmean(x) for x in weekly_bias_stations])

print('Weekly bias mean value across stations is ', np.nanmean(weekly_mean), np.nanstd(weekly_mean))

ind_week_sort = np.argsort(weekly_mean)

weeks_stations = np.array([np.unique(x) for x in M[:, 1]])

stations_names = np.array([np.unique(x) for x in M[:, 2]]).flatten()
stations_ids = np.array([np.unique(x) for x in M[:, 3]]).flatten()

print(M[:, 3], stations_ids.shape, stations_names.shape)
# stations_indicies = np.array([np.unique(x) for x in M[:, 3]])

print(weekly_bias_stations[ind_week_sort[0]])


fig, ax1 = plt.subplots(figsize=(18,7))

levels = [-16, -12, -8, -6, -2, 0, 2, 6, 8, 12, 16]
num_clrs=10
# vmin1 = np.nanmin(weekly_bias_stations)
# vmax1 = np.nanmax(weekly_bias_stations)
vmin1 = -16
vmax1 = 16
norm1, cmap1 = custom_cb(
    levels,

    num_clrs,
#    colormap = 'seismic'
)
labels_y = stations_names[ind_week_sort].astype('U80')

for i, station in enumerate(stations_names):
    # print(station, np.shape(weeks_stations[i]), np.repeat(i, weeks_stations[i].size).shape, np.shape(weekly_bias_stations[i]))
    # ind = np.repeat(i, weeks_stations[i].size)
    # x, y = np.meshgrid(weeks_stations[i], ind)
    # z = np.repeat(weekly_bias_stations[i], (ind.size))
    # print(np.shape())
    y = [i, i+1]
    x = weeks_stations[ind_week_sort[i]].tolist()
    x.append(x[-1] + 1)


    mask_z = np.isfinite(weekly_bias_stations[ind_week_sort[i]])

    z_oned = (weekly_bias_stations[ind_week_sort[i]])[mask_z]
    z = np.reshape((z_oned).tolist(), (len(y)- 1, len(x)-1))
    
    # print(labels_y[i])
    labels_y[i] += str(stations_ids[ind_week_sort[i]]) + ': '+ str(round(np.nanmean(z.flatten()), 2))
     #if 'Universität Zürich Irchel' in labels_y[i]:
     #    print('Debug', weekly_bias_stations[ind_week_sort[i]])
      #   print('Debug Station, Bias: ', labels_y[i], np.nanmean(z.flatten()))
      #   print('DEBUG z', z)
    # print(x, y, z)
    p = ax1.pcolormesh(x, y, z, cmap = cmap1, norm= norm1)


min_week = np.nanmin(np.hstack(weeks_stations))
max_week = np.nanmax(np.hstack(weeks_stations))
ax1.set_xlim([min_week, max_week])
weeks1 = np.arange(min_week, max_week+1)
year = 2022

# date = 

weeks_months = np.array([(datetime.date(2022, 1, 1) + relativedelta(weeks=+int(x))).month for x in weeks1])
weeks_months_ticks_labels = np.array([(calendar.month_name[weeks_months[np.where(weeks_months == x)[0][0]]], weeks1[np.where(weeks_months == x)[0][0]]) for x in np.unique(weeks_months)])


ax1.set_yticks(np.arange(len(stations_names))+0.5)
ax1.set_yticklabels(labels_y)

# ax1.tick_params(axis='y', which='major', labelsize=7)

ax1.set_xticks(weeks_months_ticks_labels[:, 1].astype(int))

ax1.set_xticklabels(weeks_months_ticks_labels[:, 0])
# lev_step = 4
ax1.set_title('Weekly CO2 Bias July - December 2022, ZH, ' + str(lim_0) + ':00 - ' + str(lim_1) + ':00 PM')
cbar = plt.colorbar(
    
    p, extend="both", orientation = 'vertical', shrink=0.65, pad = 0.033,
    ticks=levels,
    ax=ax1
)

# y = [0, 1]
# x = [26, 27, 28, 29, 30, 31]
# z = [4.034224305025933, 6.613080302190302, 7.89780753640251, 9.745863168431157, 8.928705224648837]
# z = np.reshape(z, (1, 5))
# ax.pcolormesh(x, y, z)


# name = MYDIR + '/WeeklyBiasZH_Damping_12_25km_22.png'
name = MYDIR + '/WeeklyBiasZH_Valley_22' + suffix + '.png'
print('Plotted: ', name)
plt.savefig(name, dpi=300, bbox_inches = 'tight')
#plt.show()
plt.clf()

