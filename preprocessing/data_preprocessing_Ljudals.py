
import os
from pathlib import Path 
import datetime

from imageio import imread, imsave

def date_to_doy(date):
    dt = datetime.datetime.strptime(date, '%Y%m%d')
    return dt.timetuple().tm_yday

def get_dateRange(start_day, end_day):
    start = datetime.datetime.strptime(start_day, "%Y%m%d")
    end = datetime.datetime.strptime(end_day, "%Y%m%d")
    date_range = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    return date_range 



eventPath = Path("E:\PyProjects/temporal-consistency\data\Ljusdals")
firms_full_path = eventPath / "viirs_full"
firmsDateList = [firmsName[:8] for firmsName in sorted(os.listdir(firms_full_path))]

import tifffile as tiff
modis = tiff.imread(eventPath / "modisPrgMap.tif")
fusion_maskDir = eventPath / "fusion_full"
if not os.path.exists(fusion_maskDir): os.makedirs(fusion_maskDir)

## Fill VIIRS gap dates
if False:
    for date in get_dateRange(firmsDateList[0], "20180810"): #viirsList[-1][:8]
        date = date.strftime("%Y%m%d")

        print(date)

        firmsName = f"{date}_firms.B0.png"
        if os.path.isfile(firms_full_path / firmsName):
            viirs = imread(firms_full_path / firmsName)

        else:
            
            imsave(firms_full_path / firmsName, viirs)

        
        julian_day = date_to_doy(date)
        modis_base = ((modis <= julian_day) & (modis > 0)).astype(float)

        modviirs = ((modis_base + viirs) > 0).astype(float)
        imsave(fusion_maskDir / firmsName, modviirs)


## Assign S2 labelMask
if True:
    dataFolder = "s2_data"

    for folder in ['viirs_full', 'fusion_full']:
        firms_full_path = eventPath / folder

        sentinel2_dataDir = eventPath / dataFolder

        sentinel2_maskDir = eventPath / f"{dataFolder}_mask"
        if 'fusion' in os.path.split(firms_full_path)[-1]:
            sentinel2_maskDir = eventPath / f"{dataFolder}_mask_fusion" 
            
        if not os.path.exists(sentinel2_maskDir): os.makedirs(sentinel2_maskDir)

        for filename in sorted(os.listdir(sentinel2_dataDir)):
            print(filename)

            if 's2' in os.path.split(sentinel2_dataDir)[-1]:
                s2_date = filename.split("_")[0][:8]
            
            if 's1' in os.path.split(sentinel2_dataDir)[-1]:
                s2_date = filename.split("_")[0][:8]
                # s2_date = filename.split("_")[1][:8]

            print("s2_date: ", s2_date)

            if s2_date < firmsDateList[0]:
                mask = imread(firms_full_path / f"{firmsDateList[0]}_firms.B0.png")

            elif s2_date > firmsDateList[-1]:
                mask = imread(firms_full_path / f"{firmsDateList[-1]}_firms.B0.png")
            
            else: 
                mask = imread(firms_full_path / f"{s2_date}_firms.B0.png")

            imsave(sentinel2_maskDir / filename, mask)
        
    








