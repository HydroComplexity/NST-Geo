import numpy as np
import pandas as pd
from tifffile import imread
import torch


######################  weather data #####################
Weath_add='/home/.../weather data monticello/'
Wdataread2022=pd.read_excel(Weath_add+'S-153-SRFP2_2022_Aggregate.xlsx')
Wdataread2023=pd.read_excel(Weath_add+'S-153-SRFP2_2023_Aggregate.xlsx')


Wdataread2022=Wdataread2022.iloc[2:,:].values
Wdataread2023=Wdataread2023.iloc[1:,:].values

Wdata2022=Wdataread2022[:,[0,1,3,4]]  #time, ppt, temp, SR
Wdata2023=Wdataread2023[:,[0,2,3,6]] #time, ppt, temp, SR

datetime_series = pd.to_datetime(Wdata2023[:,0])
datetime_series=pd.Series(datetime_series)
datetime_plus_5=datetime_series+pd.Timedelta(hours=4, minutes=59)
datetime_without_tz = datetime_plus_5.dt.tz_localize(None)
dt_index_no_tz = pd.to_datetime(datetime_without_tz)
Wdata2023[:,0]=dt_index_no_tz

###degree days 10oC as min temp for plant growth
Wdata2022[:,2],Wdata2023[:,2]=Wdata2022[:,2]-10,Wdata2023[:,2]-10
Wdata2022[Wdata2022[:,2]<0,2]=0
Wdata2023[Wdata2023[:,2]<0,2]=0

### aggrigating the data to 1 day frequency
## 2022
df1=pd.DataFrame(Wdata2022[:,1], index=Wdata2022[:,0], columns=['0'])
df_positive1 = df1[df1['0'] >= 0]
df_daily_pptsum22 = df_positive1.resample('D').sum()

df2=pd.DataFrame(Wdata2022[:,2], index=Wdata2022[:,0], columns=['1'])
df_positive2 = df2[df2['1'] > 0]
df_daily_DDmean22 = df_positive2.resample('D').mean()

df3=pd.DataFrame(Wdata2022[:,3], index=Wdata2022[:,0], columns=['2'])
df_positive3 = df3[df3['2'] > 0]
df_daily_SRmean22 = df_positive3.resample('D').mean()

### 2023
df1=pd.DataFrame(Wdata2023[:,1], index=Wdata2023[:,0], columns=['0'])
df_positive1 = df1[df1['0'] >= 0]
df_daily_pptsum23 = df_positive1.resample('D').sum()

df2=pd.DataFrame(Wdata2023[:,2], index=Wdata2023[:,0], columns=['1'])
df_positive2 = df2[df2['1'] > 0]
df_daily_DDmean23 = df_positive2.resample('D').mean()

df3=pd.DataFrame(Wdata2023[:,3], index=Wdata2023[:,0], columns=['2'])
df_positive3 = df3[df3['2'] > 0]
df_daily_SRmean23 = df_positive3.resample('D').mean()


##### spatial data encoder NDVI data  ##########################
def get_ndvidata():
    ndvipath='/home/.../ndvi/'
    banddata230211=imread(ndvipath+'20230211_3bands_clip.tif')
    banddata230115=imread(ndvipath+'20230115_3bands_clip.tif')
    banddata221126=imread(ndvipath+'20221126_3bands_clip.tif')
    banddata221103=imread(ndvipath+'20221103_3bands_clip.tif')
    banddata221003=imread(ndvipath+'20221003_3bands_clip.tif')
    banddata220918=imread(ndvipath+'20220918_3bands_clip.tif')
    banddata220818=imread(ndvipath+'20220818_3bands_clip.tif')
    banddata220722=imread(ndvipath+'20220722_3bands_clip.tif')
    banddata220703=imread(ndvipath+'20220703_3bands_clip.tif')
    banddata220618=imread(ndvipath+'20220618_3bands_clip.tif')
    banddata220523=imread(ndvipath+'20220523_3bands_clip.tif')
    banddata220507=imread(ndvipath+'20220507_3bands_clip.tif')
    banddata220403=imread(ndvipath+'20220403_3bands_clip.tif')
    banddata230315=imread(ndvipath+'20230315_3bands_clip.tif')

    banddata230211=banddata230211.astype(float)
    banddata230115=banddata230115.astype(float)
    banddata221126=banddata221126.astype(float)
    banddata221103=banddata221103.astype(float)
    banddata221003=banddata221003.astype(float)
    banddata220918=banddata220918.astype(float)
    banddata220818=banddata220818.astype(float)
    banddata220722=banddata220722.astype(float)
    banddata220703=banddata220703.astype(float)
    banddata220618=banddata220618.astype(float)
    banddata220523=banddata220523.astype(float)
    banddata220507=banddata220507.astype(float)
    banddata220403=banddata220403.astype(float)
    banddata230315=banddata230315.astype(float)



    mask230211=(banddata230211[:,:,3]+banddata230211[:,:,0])==0
    ndvi230211=np.where(mask230211, np.nan, (banddata230211[:,:,3]-banddata230211[:,:,0])/(banddata230211[:,:,3]+banddata230211[:,:,0]))
    mask230115=(banddata230115[:,:,3]+banddata230115[:,:,0])==0
    ndvi230115=np.where(mask230115, np.nan, (banddata230115[:,:,3]-banddata230115[:,:,0])/(banddata230115[:,:,3]+banddata230115[:,:,0]))
    mask221126=(banddata221126[:,:,3]+banddata221126[:,:,0])==0
    ndvi221126=np.where(mask221126, np.nan, (banddata221126[:,:,3]-banddata221126[:,:,0])/(banddata221126[:,:,3]+banddata221126[:,:,0]))
    mask221103=(banddata221103[:,:,3]+banddata221103[:,:,0])==0
    ndvi221103=np.where(mask221103, np.nan, (banddata221103[:,:,3]-banddata221103[:,:,0])/(banddata221103[:,:,3]+banddata221103[:,:,0]))
    mask221003=(banddata221003[:,:,3]+banddata221003[:,:,0])==0
    ndvi221003=np.where(mask221003, np.nan, (banddata221003[:,:,3]-banddata221003[:,:,0])/(banddata221003[:,:,3]+banddata221003[:,:,0]))
    mask220918=(banddata220918[:,:,3]+banddata220918[:,:,0])==0
    ndvi220918=np.where(mask220918, np.nan, (banddata220918[:,:,3]-banddata220918[:,:,0])/(banddata220918[:,:,3]+banddata220918[:,:,0]))
    mask220818=(banddata220818[:,:,3]+banddata220818[:,:,0])==0
    ndvi220818=np.where(mask220818, np.nan, (banddata220818[:,:,3]-banddata220818[:,:,0])/(banddata220818[:,:,3]+banddata220818[:,:,0]))
    mask220722=(banddata220722[:,:,3]+banddata220722[:,:,0])==0
    ndvi220722=np.where(mask220722, np.nan, (banddata220722[:,:,3]-banddata220722[:,:,0])/(banddata220722[:,:,3]+banddata220722[:,:,0]))
    mask220703=(banddata220703[:,:,3]+banddata220703[:,:,0])==0
    ndvi220703=np.where(mask220703, np.nan, (banddata220703[:,:,3]-banddata220703[:,:,0])/(banddata220703[:,:,3]+banddata220703[:,:,0]))
    mask220618=(banddata220618[:,:,3]+banddata220618[:,:,0])==0
    ndvi220618=np.where(mask220618, np.nan, (banddata220618[:,:,3]-banddata220618[:,:,0])/(banddata220618[:,:,3]+banddata220618[:,:,0]))
    mask220523=(banddata220523[:,:,3]+banddata220523[:,:,0])==0
    ndvi220523=np.where(mask220523, np.nan, (banddata220523[:,:,3]-banddata220523[:,:,0])/(banddata220523[:,:,3]+banddata220523[:,:,0]))
    mask220507=(banddata220507[:,:,3]+banddata220507[:,:,0])==0
    ndvi220507=np.where(mask220507, np.nan, (banddata220507[:,:,3]-banddata220507[:,:,0])/(banddata220507[:,:,3]+banddata220507[:,:,0]))
    mask220403=(banddata220403[:,:,3]+banddata220403[:,:,0])==0
    ndvi220403=np.where(mask220403, np.nan, (banddata220403[:,:,3]-banddata220403[:,:,0])/(banddata220403[:,:,3]+banddata220403[:,:,0]))
    mask230315=(banddata230315[:,:,3]+banddata230315[:,:,0])==0
    ndvi230315=np.where(mask230315, np.nan, (banddata230315[:,:,3]-banddata230315[:,:,0])/(banddata230315[:,:,3]+banddata230315[:,:,0]))




    return ndvi230211, ndvi230115, ndvi221126, ndvi221103, ndvi221003, ndvi220918, ndvi220818, ndvi220722, ndvi220703, ndvi220618, ndvi220523, ndvi220507, ndvi220403, ndvi230315
################### porosity data  ###################
def get_porosity():
    catchmentpath='/home/.../soil_area/'
    porosity=imread(catchmentpath+'porosity.tif')
    porosity[porosity[:,:]<-1]=np.nan
    porosity[porosity[:,:]>1]=np.nan
    return porosity
################### slope data  ###################
def get_slope():
    catchmentpath='/home/.../soil_area/'
    slope=imread(catchmentpath+'slope.tif')
    slope[slope[:,:]<-10000]=np.nan
    slope[slope[:,:]>10000]=np.nan
    return slope
################### aspect data  ###################
def get_aspect():
    catchmentpath='/home/.../soil_area/'
    aspect=imread(catchmentpath+'Aspect_dem_sgeodesic_2.tif')
    aspect[aspect[:,:]<-10000]=np.nan
    aspect[aspect[:,:]>361]=np.nan
    return aspect
################### area data  ###################
def get_area():
    catchmentpath='/home/.../soil_area/'
    area=imread(catchmentpath+'area.tif')
    area[area[:,:]<-10000]=np.nan
    area[area[:,:]>10000]=np.nan
    return area



################### Autoencoder model  ###################

def masked_mse_loss(output, target, mask):
    # Compute MSE only for valid (non-NaN) values
    diff = (output - target) ** 2
    masked_diff = diff * mask  # Apply mask
    loss = masked_diff.sum() / mask.sum()  # Normalize by the number of valid entries
    return loss

def mask_nan(x):
    # Replace NaN values with 0 or mean (for temporary input)
    xmask = ~torch.isnan (x)  #
    x[~xmask] = 0  # Replace NaN with 0
    return x, xmask


def min_max_scaler_nan(data) :
    """
    Scales a 2D numpy array to the range [0, 1] while preserving NaN values.

    Args:
        data (numpy.ndarray): The 2D numpy array to scale.

    Returns:
        numpy.ndarray: The scaled 2D numpy array.
    """
    data_copy = np.copy (data).astype (float)
    nan_mask = np.isnan (data_copy)
    min_val = np.nanmin (data_copy)
    max_val = np.nanmax (data_copy)

    if min_val == max_val :
        data_copy[~nan_mask] = 0.0
    else :
        data_copy[~nan_mask] = (data_copy[~nan_mask] - min_val) / (max_val - min_val)

    return data_copy


def minmax_scale_tensor(data, feature_range=(0, 1)) :
    min_val, max_val = feature_range
    data_min = torch.min (data)
    data_max = torch.max (data)
    # Normalize
    normalized_data = (data - data_min) / (data_max - data_min)
    normalized_data = normalized_data * (max_val - min_val) + min_val

    return normalized_data


