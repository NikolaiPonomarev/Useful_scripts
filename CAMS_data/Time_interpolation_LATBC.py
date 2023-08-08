import xarray as xr
import numpy as np
import os
import datetime
from datetime import timedelta
from multiprocessing import Pool
import glob 

# startdate = datetime.datetime(2022, 7, 1, 0)
startdate = datetime.datetime(2022, 10, 25, 0)
#enddate = datetime.datetime(2022, 7, 31, 23)
enddate = datetime.datetime(2023, 2, 26, 23)
# enddate = datetime.datetime(2023, 10, 24, 23)
delta = enddate - startdate
chosen_dates = []
for i in range(delta.days + 1):
    
    day = startdate + timedelta(days=i)
    for h in range(0,24, 1):
        day1 = day + timedelta(hours=h)
        chosen_dates.append(day1)


chosen_dates.append(day1 + + timedelta(hours=1))
print(chosen_dates)


# 2022090212.nc
# path = '/store/empa/em05/dbrunner/paul/icbc/cams_hlkx_'
pattern = 'cams_hues_'
# pattern = 'cams_hlkx_'

outfile = pattern + 'merged_tmp.nc'
# g_files = [path + x.strftime('%Y%m%d%H')+'.nc' for x in chosen_dates]

#Merge data into a single file wiith all timesteps
os.system('cdo -b F64 mergetime /store/empa/em05/dbrunner/paul/icbc/' + pattern + '*.nc ' + outfile)

#Temporal interpolation
# cdo inttime,1987-01-01,12:00:00,1hour infile outfile
startdate = '2022-10-25,00:00:00'
# startdate = '2022-07-01,00:00:00'
outfile_interpolated = 'cams_hues_merged_interpolated.nc'
# outfile_interpolated = 'cams_hlkx_merged_interpolated.nc'

command = 'cdo -b F64 inttime,' + startdate + ',1hour ' + outfile + ' ' + outfile_interpolated 
print('Interpolating using the following command: ', command)
os.system(command)

pattern_hlkx = 'cams_hlkx_'
# #Split one big file with interpolated data into hourly files
output_dir = '/scratch/snx3000/nponomar/processing_chain_python/processing-chain/work/CAMS_hourly_data_interpolated_Dominik/'
command_split = 'cdo -splitsel,1 ' + outfile_interpolated + ' ' + output_dir + pattern_hlkx#pattern
print('Split one big file into hourly files with the following command: ', command_split)
os.system(command_split)
pattern_hlkx_0 = 'cams_hlkx_0'
#Move files to have dates in their names

files = sorted(glob.glob(output_dir + pattern_hlkx_0 + "*")) #get the list of files
# print(files)
for ix, x in enumerate(chosen_dates):
    infile = files[ix]
    outputfile = output_dir + pattern_hlkx + x.strftime("%Y%m%d%H")#pattern
    
    command_mv = 'mv ' + infile + ' ' + outputfile
    print(command_mv)
    os.system(command_mv)

