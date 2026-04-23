# Standard PyTorch imports
import numpy as np
import torch
import NST_functions as nstf
import pickle
import pandas as pd



def readpickle(address, file_name):
    with open(address + file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def remove_nan_rows_cols(arr) :
    """Remove entirely NaN rows and columns from 2D array"""
    if arr.size == 0 :
        return arr

    # Remove entirely NaN rows
    arr_clean = arr[~np.isnan (arr).all (axis=1)]

    # Remove entirely NaN columns (check if array still has data)
    if arr_clean.size > 0 :
        arr_clean = arr_clean[:, ~np.isnan (arr_clean).all (axis=0)]

    return arr_clean


def coarsen_2d_preserve_nan(arr, factor=2) :
    """
    Coarsen array where:
    - If a block contains ANY NaN, the result is NaN
    - If a block has no NaN, compute mean normally
    """
    m, n = arr.shape
    m_new = (m // factor) * factor
    n_new = (n // factor) * factor

    # Trim array if necessary
    arr_trimmed = arr[:m_new, :n_new]

    # Reshape into blocks
    reshaped = arr_trimmed.reshape (m_new // factor, factor,
                                    n_new // factor, factor)

    # Check for NaN in each block
    has_nan = np.isnan (reshaped).any (axis=(1, 3))

    # Compute mean for blocks without NaN
    coarsened = np.nanmean (reshaped, axis=(1, 3))

    # Set blocks with NaN to NaN
    coarsened[has_nan] = np.nan

    return coarsened


def coarsen_2d_exclude_nan_from_mean(arr, factor=2) :
    """
    For each block:
    - Compute mean only from non-NaN values
    - If all values in block are NaN, result is NaN
    - If some values are NaN, compute mean from remaining values
    """
    m, n = arr.shape
    m_new = (m // factor) * factor
    n_new = (n // factor) * factor

    # Trim array
    arr_trimmed = arr[:m_new, :n_new]

    # Reshape into blocks
    reshaped = arr_trimmed.reshape (m_new // factor, factor,
                                    n_new // factor, factor)

    # Use nanmean (ignores NaN in calculation)
    coarsened = np.nanmean (reshaped, axis=(1, 3))

    return coarsened


def coarsen_dict_preserve_nan(arrays_dict, factor=2, method='preserve', custom_factors=None) :

    if custom_factors is None :
        custom_factors = {}

    if method == 'preserve' :
        coarsen_func = coarsen_2d_preserve_nan
    else :
        coarsen_func = coarsen_2d_exclude_nan_from_mean

    result = {}
    for key, arr in arrays_dict.items () :
        # Use custom factor if specified, otherwise use default
        current_factor = custom_factors.get (key, factor)
        result[key] = coarsen_func (arr, current_factor)

    return result

# 30 min data

address='/home/.../monticello_data_2223/'

gi1,gf1=1,17400


data = pd.read_csv(address+'monticello_0.5hr.csv')
data=data.iloc[gi1:gf1,1:].values

static_datard1 = readpickle ('/home/.../','filename.pickle')
keys = list (static_datard1.keys ())
values = list (static_datard1.values ())
cleaned_values = list (map (remove_nan_rows_cols, values))
static_datard2 = dict (zip (keys, cleaned_values))


scalefactor=[]
static_datard = coarsen_dict_preserve_nan (static_datard2, factor=scalefactor, method='preserve')  #custom_factors={'ndvi' : } if any

###########################################################
###########################################################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

staticid='yes'      #yes,no,random
slopeid='no'        #yes: only slope will be randomized
aspectid='no'       #yes: only aspect will be randomized
porosityid='no'     #yes: only porosity will be randomized
ndviid='yes'        #yes,no,random
weatherid='no'      #we dont use weather data in this model



###### overall testing predicting from all solutes to all solutes ##############
prediction_horizon=[30,40,50,100,150]
batch_size=[200,200,200,200,200]
history_len=[5,10,20,30,40,50,60,70,80,100,125,150]


column_names = np.array (['Ca', 'Cl', 'Mg', 'NO3', 'K', 'Na', 'SO4', 'Q'])
var_list = ['Ca', 'Cl', 'Mg', 'NO3', 'K', 'Na', 'SO4', 'Q']


def train_model_file(file,static_datard,column_names,varn,prediction_horizon,batch_size,history_len,staticid,ndviid,weatherid,slopeid,aspectid,porosityid,temp_shape) :
    data = pd.DataFrame (file, columns=column_names)
    feature_length = data.shape[1]
    history_length = history_len   #150 before
    normalization = 'max-min'
    print('Now starting the training of the model for {} with prediction horizon {} and history length {} and batch size {}'.format (varn, prediction_horizon, history_length, batch_size))
    model, outputs = nstf.fit_non_stationary_transformer (data,static_datard,staticid,ndviid,weatherid,slopeid,aspectid,porosityid,train_fr=0.7,feature_length=feature_length, history_length=history_length,
                                                     prediction_horizon=prediction_horizon,
                                                     N=1, d_model=32, d_ff=128, h=8, dropout=0.1, epochs=30,
                                                     batch_size=batch_size, #1280
                                                     flow_dr='yes',savefig=True, showfig=False, savevars=True,
                                                     saveloc="/home/.../{}_random_{}_predicting_{}_{}".format ('r3',history_length,
                                                                                                     prediction_horizon,
                                                                                                     normalization),
                                                     device=device, normalization=normalization)

    return model, outputs



for i in range(len(prediction_horizon)):
    for j in range(len(history_len)):
        train_model_file (data,static_datard,column_names, var_list,prediction_horizon[i], batch_size[i], history_len[j],staticid,ndviid,weatherid,slopeid,aspectid,porosityid,10)  #all solute to all solute learning and prediction
