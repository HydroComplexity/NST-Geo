import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import utils_p3
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from matplotlib.colors import ListedColormap


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

def load_and_reconstruct_params(pickle_path) :
    """
    Load saved variables and reconstruct parameter objects
    """
    with open (pickle_path, 'rb') as handle :
        loaded_vars = pickle.load (handle)

    config = loaded_vars.get ('Config', {})
    staticid = config.get ('staticid', 'no')
    ndviid = config.get ('ndviid', 'no')

    print (f"Loaded model with staticid={staticid}, ndviid={ndviid}")

    # Reconstruct model
    model = loaded_vars['Model']

    # Reconstruct static parameters if they exist
    static_params = {}
    if 'Static_param' in loaded_vars :
        print ("Found static parameters:")
        for param_name, state_dict in loaded_vars['Static_param'].items () :
            print (f"  - {param_name}")
            # You'll need to recreate the parameter objects with same architecture
            # Example: static_params[param_name] = YourParameterClass()
            # static_params[param_name].load_state_dict(state_dict)

    # Reconstruct NDVI parameters if they exist
    ndvi_params = {}
    if 'NDVI_param' in loaded_vars :
        print ("Found NDVI parameters:")
        for param_name, state_dict in loaded_vars['NDVI_param'].items () :
            print (f"  - {param_name}")
            # You'll need to recreate the parameter objects with same architecture
            # Example: ndvi_params[param_name] = YourNDVIParameterClass()
            # ndvi_params[param_name].load_state_dict(state_dict)

    return loaded_vars, static_params, ndvi_params


def nan_helper(y) :
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan (y), lambda z : z.nonzero ()[0]

def heatmap(data,cmap,var,ph,hl,title,err,plotadd,save='no'):

    ax = sns.heatmap (data,cmap=cmap,xticklabels=hl,yticklabels=ph,vmax=1,vmin=0,linewidth=0.5,cbar_kws={'label': 'NSE'})

    ax.set_xlabel('History Length',fontsize=12)
    ax.set_ylabel('Prediction horizon',fontsize=12)
    plt.title (title+err+' '+var,fontsize=14)
    ax.figure.axes[-1].yaxis.label.set_size (13)
    if save=='yes':
        plt.savefig(plotadd+var+'_'+title+'_heatmap.png')
    plt.show ()

def heatmap1(data, cmap, var, ph, hl, title, err,diff, plotadd, save='no') :
    dmax=np.max(data)

    ax = sns.heatmap (data, cmap=cmap, xticklabels=hl, yticklabels=ph,vmax=dmax,vmin=-dmax,linewidth=0.5,
                      cbar_kws={'label' : '$\Delta$(NSE)'})

    ax.set_xlabel ('History Length', fontsize=12)
    ax.set_ylabel ('Prediction horizon', fontsize=12)
    plt.title (title + err + ' ' + var, fontsize=14)
    ax.figure.axes[-1].yaxis.label.set_size (13)
    if save == 'yes' :
        plt.savefig (plotadd + var + '_' + diff + '_heatmap.png')
    plt.show ()

def intp(X) :
    training_set0 = np.empty (shape=X.shape)
    xx = X.astype (np.cfloat)
    for i in range (X.shape[1]) :
        xx1 = xx[:, i]
        nans, yy = nan_helper (xx1)
        xx1[nans] = np.interp (yy (nans), yy (~nans), xx1[~nans])
        training_set0[:, i] = xx1
    return training_set0


def calculate_nse(observed, predicted) :
    """
    Calculate Nash-Sutcliffe Efficiency
    NSE = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
    """
    observed = np.array (observed)
    predicted = np.array (predicted)

    # Calculate mean of observed values
    obs_mean = np.mean (observed)

    # Calculate NSE
    numerator = np.sum ((observed - predicted) ** 2)
    denominator = np.sum ((observed - obs_mean) ** 2)

    if denominator == 0 :
        return np.nan

    nse = 1 - (numerator / denominator)
    return nse


def kge_fn(obs, pred) :

    pred = np.array (pred)
    obs = np.array (obs)

    mask = ~np.isnan (obs) & ~np.isnan (pred)
    pred = pred[mask]
    obs = obs[mask]

    cc = np.corrcoef (pred, obs)[0, 1]
    alpha = np.std (pred) / np.std (obs)
    beta = np.mean (pred) / np.mean (obs)

    kge = 1 - np.sqrt ((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge

#calculated the area under nse-cdf curve while giving the observed and predicted values of the model
def nse_cdf_area(observed, predicted) :
    """
    Simple function to calculate area under NSE-CDF curve
    """
    # Convert to numpy arrays
    obs = np.array (observed)
    pred = np.array (predicted)

    # Calculate NSE
    # obs_mean = np.mean (obs)
    # nse = 1 - np.sum ((obs - pred) ** 2) / np.sum ((obs - obs_mean) ** 2)

    # Create multiple NSE values by sampling different portions of data
    nse_values = []
    n = len (obs)

    # Sample different portions of the data
    for i in range (10, n, 5) :  # Start from 10 points, step by 5
        sample_obs = obs[:i]
        sample_pred = pred[:i]
        sample_mean = np.mean (sample_obs)
        sample_nse = 1 - np.sum ((sample_obs - sample_pred) ** 2) / np.sum ((sample_obs - sample_mean) ** 2)
        nse_values.append (sample_nse)

    # Sort NSE values
    nse_values = np.sort (nse_values)

    # Create CDF (cumulative probabilities)
    cdf = np.arange (1, len (nse_values) + 1) / len (nse_values)

    # Calculate area under curve
    area = np.trapz (cdf, nse_values)

    return area#, nse_values, cdf, nse


def cumulative_error(y_true, y_pred):
    length=len(y_true)
    avg=np.mean(y_true,axis=0)
    abs_error = np.abs(y_true - y_pred)
    return np.sum(abs_error)/(avg*length)

def predobs_rmse_nse_pickle(prediction_horizon,history_len,address,f_drid):
    ph = prediction_horizon
    hl = history_len
    nse = np.empty ((8))
    kge = np.empty ((8))
    nse_area = np.empty ((8))
    cme = np.empty ((8))
    if f_drid=='no':
        file_name_fdr = '{}_predicting_{}_max-min_data_.pickle'.format(hl,ph)
    else:
        file_name_fdr = '{}_predicting_{}_max-min_data_flow_dr_.pickle'.format (hl, ph)

    data_fdr = readpickle (address, file_name_fdr)
    pred_fdr = data_fdr['Predictions']
    obs_fdr = data_fdr['Truth']
    obs1_fdr = obs_fdr[:len (pred_fdr)]
    ### tensor to numpy
    obs1_fdr = np.array (obs1_fdr)
    pred_fdr = np.array (pred_fdr)
    if np.isnan (pred_fdr[:, :7]).all () == False :
        obs1_fdr = intp (obs1_fdr)
        pred_fdr = intp (pred_fdr)
        rmse = np.sqrt (mean_squared_error (obs1_fdr[:5000], pred_fdr[:5000], multioutput='raw_values'))
        # cme=cumulative_error(obs1_fdr[:5000], pred_fdr[:5000])
        for k in range (7) :
            cme[k] = cumulative_error (obs1_fdr[:5000,k], pred_fdr[:5000,k])
            nse[k] = calculate_nse(obs1_fdr[:5000, k], pred_fdr[:5000, k])
            kge[k] = kge_fn(obs1_fdr[:5000, k], pred_fdr[:5000, k])
            nse_area[k] = nse_cdf_area(obs1_fdr[:5000, k], pred_fdr[:5000, k])
    else :
        rmse = np.nan
        print ('prediction contains nan values, skipping this prediction horizon and history length')

    return obs1_fdr, pred_fdr, rmse, nse,nse_area,cme,kge

def predobs_rmse_nse_pickle_rd(prediction_horizon,history_len,address,f_drid,rd):
    ph = prediction_horizon
    hl = history_len
    nse = np.empty ((8))
    nse_area = np.empty ((8))
    if f_drid=='no':
        file_name_fdr = '{}_random_{}_predicting_{}_max-min_data_.pickle'.format(rd,hl,ph)
    else:
        file_name_fdr = '{}_random_{}_predicting_{}_max-min_data_flow_dr_.pickle'.format (rd,hl, ph)

    data_fdr = readpickle (address, file_name_fdr)
    pred_fdr = data_fdr['Predictions']
    obs_fdr = data_fdr['Truth']
    obs1_fdr = obs_fdr[:len (pred_fdr)]
    ### tensor to numpy
    obs1_fdr = np.array (obs1_fdr)
    pred_fdr = np.array (pred_fdr)
    if np.isnan (pred_fdr[:, :7]).all () == False :
        obs1_fdr = intp (obs1_fdr)
        pred_fdr = intp (pred_fdr)
        rmse = np.sqrt (mean_squared_error (obs1_fdr[:5000], pred_fdr[:5000], multioutput='raw_values'))
        cme=cumulative_error(obs1_fdr[:5000], pred_fdr[:5000])
        for k in range (7) :
            nse[k] = calculate_nse(obs1_fdr[:5000, k], pred_fdr[:5000, k])

            nse_area[k] = nse_cdf_area(obs1_fdr[:5000, k], pred_fdr[:5000, k])
    else :
        rmse = np.nan
        cme=np.nan
        print ('prediction contains nan values, skipping this prediction horizon and history length')

    return obs1_fdr, pred_fdr, rmse, nse,nse_area,cme


def param_predobs_rmse_nse_pickle(prediction_horizon,history_len,address,f_drid,res):
    ph = prediction_horizon
    hl = history_len
    nse = np.empty ((8))
    nse_area = np.empty ((8))
    if f_drid=='no':
        file_name_fdr = '{}_predicting_{}_max-min_shape_{}_data_.pickle'.format(hl,ph,res)
    else:
        file_name_fdr = '{}_predicting_{}_max-min_shape_{}_data_flow_dr_.pickle'.format (hl, ph,res)

    data_fdr = readpickle (address, file_name_fdr)
    pred_fdr = data_fdr['Predictions']
    obs_fdr = data_fdr['Truth']
    obs1_fdr = obs_fdr[:len (pred_fdr)]
    ### tensor to numpy
    obs1_fdr = np.array (obs1_fdr)
    pred_fdr = np.array (pred_fdr)
    if np.isnan (pred_fdr[:, :7]).all () == False :
        obs1_fdr = intp (obs1_fdr)
        pred_fdr = intp (pred_fdr)
        rmse = np.sqrt (mean_squared_error (obs1_fdr[:5000], pred_fdr[:5000], multioutput='raw_values'))
        for k in range (7) :
            nse[k] = calculate_nse(obs1_fdr[:5000, k], pred_fdr[:5000, k])
            nse_area[k] = nse_cdf_area(obs1_fdr[:5000, k], pred_fdr[:5000, k])
    else :
        rmse = np.nan
        print ('prediction contains nan values, skipping this prediction horizon and history length')

    return obs1_fdr, pred_fdr, rmse, nse,nse_area

def cc_predobs_rmse_nse_pickle(varn,prediction_horizon,history_len,address,f_drid):
    ph = prediction_horizon
    hl = history_len
    if f_drid=='no':
        file_name_fdr = '{}_var_{}_predicting_{}_max-min_data_.pickle'.format(varn,hl,ph)
    else:
        file_name_fdr = '{}_var_{}_predicting_{}_max-min_data_flow_dr_.pickle'.format (varn,hl, ph)

    data_fdr = readpickle (address, file_name_fdr)
    pred_fdr = data_fdr['Predictions']
    obs_fdr = data_fdr['Truth']
    obs1_fdr = obs_fdr[:len (pred_fdr)]
    ### tensor to numpy
    obs1_fdr = np.array (obs1_fdr)
    pred_fdr = np.array (pred_fdr)
    if np.isnan (pred_fdr[:, :1]).all () == False :
        obs1_fdr = intp (obs1_fdr)
        pred_fdr = intp (pred_fdr)
        rmse = np.sqrt (mean_squared_error (obs1_fdr[:5000], pred_fdr[:5000], multioutput='raw_values'))
        # nse = calculate_nse (obs1_fdr[:5000, 0], pred_fdr[:5000, 0])
    else :
        rmse = np.nan
        # nse=np.nan
        print ('prediction contains nan values, skipping this prediction horizon and history length')

    return obs1_fdr, pred_fdr, rmse

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

def nse_ph(nse_nst,nse_geo,varn,ph,plotadd,save) :
    plt.figure (figsize=(8, 6))
    plt.plot (ph, nse_nst, marker='o', label='NST', color='darkorange')
    plt.plot (ph, nse_geo, marker='o', label='NST-Geo', color='#1f77b4')
    plt.xlabel ('Prediction Horizon (time steps)', fontsize=12)
    plt.ylabel ('NSE', fontsize=12)
    plt.title ('NSE vs Prediction Horizon for {}'.format (varn), fontsize=14)
    plt.xticks (ph)
    plt.ylim (-0.5, 1.2)
    # plt.grid (True)
    plt.legend (fontsize=12,loc='lower left')
    if save =='yes':
        plt.savefig (plotadd + 'nse_{}_ph_comparison.png'.format (varn))
    plt.show ()


def nse_ph_combined(nse_nst_all, nse_geo_all, var_listn1, prediction_horizon, plotadd, save='no') :

    n_solutes = len (var_listn1)

    # Create subplots (7 rows, 1 column)
    fig, axes = plt.subplots (n_solutes, 1,figsize=(6, 6), sharex=True)

    for k, ax in enumerate (axes) :
        nse_nst = nse_nst_all[:, 0, k]
        nse_geo = nse_geo_all[:, 0, k]
        varn = var_listn1[k]

        ax.scatter(prediction_horizon, nse_nst, label='NST',
                   color='darkorange', s=70,facecolors='none', edgecolors='darkorange')
        ax.scatter(prediction_horizon, nse_geo, label='NST-Geo',
                   color='darkorchid', s=30)
        ax.set_ylim (.7, 1.1)
        # ax.set_title (varn, fontsize=12, loc='left')
        ax.text(
            0.02, 0.05, varn,
            transform=ax.transAxes,
            fontsize=14,
            color='black',
            ha='left', va='bottom'
        )
        ax.grid (False)


    # Adjust layout and add one common x label
    # fig.tight_layout (pad=1)
    fig.supxlabel ('Prediction Horizon (time steps)', fontsize=12)
    fig.supylabel ('Nash-Sutcliffe Efficiency (NSE)', fontsize=12)
    fig.suptitle ('NSE vs. Prediction Horizon for All Solutes', fontsize=14)

    if save == 'yes' :
        plt.savefig (plotadd + 'nse_all_solutes_ph_comparison.png', bbox_inches='tight', dpi=300)

    plt.show ()


def nse_ph_combined1(nse_nst_all, nse_geo_all, var_listn1, prediction_horizon, plotadd, save='no') :

    n_solutes = len(var_listn1)

    # Determine number of rows for 2 columns
    ncols = 2
    nrows = int(np.ceil(n_solutes / ncols))

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True)
    axes = axes.flatten()  # flatten for easy indexing

    for k in range(n_solutes):
        ax = axes[k]
        nse_nst = nse_nst_all[:, 0, k]
        nse_geo = nse_geo_all[:, 0, k]
        varn = var_listn1[k]

        # Ensure x and y lengths match
        x = prediction_horizon if len(prediction_horizon) == len(nse_nst) else np.arange(len(nse_nst))

        # Scatter plots
        ax.scatter(x, nse_nst, label='NST', color='darkorange', s=280, facecolors='none', edgecolors='darkorange')
        ax.scatter(x, nse_geo, label='NST-Geo', color='darkorchid', s=170)

        # Set y-limits and horizontal lines
        ax.set_ylim(.7, 1.05)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        # ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Label solute
        ax.text(0.02, 0.05, varn, transform=ax.transAxes, fontsize=18, color='black', ha='left', va='bottom')

        # Grid and tick size
        ax.grid(False)
        ax.tick_params(axis='y', labelsize=12)

        # Show x-tick labels only for the bottom row plots (all columns)
        if (k + ncols >= n_solutes) :
            ax.tick_params (axis='x', labelbottom=True, labelsize=12)
        else :
            ax.tick_params (axis='x', labelbottom=False)

    # Remove empty axes if n_solutes < nrows*ncols
    for j in range(n_solutes, nrows * ncols):
        fig.delaxes(axes[j])

    # Common labels and title
    fig.supxlabel('Prediction Horizon (time steps)', fontsize=14)
    fig.supylabel('Nash-Sutcliffe Efficiency (NSE)', fontsize=14)
    fig.suptitle('NSE vs. Prediction Horizon for All Solutes', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

    # Save figure if needed
    if save == 'yes' :
        plt.savefig (plotadd + 'nse_all_solutes_ph_comparison1.png', bbox_inches='tight', dpi=300)

    plt.show ()
############## pickle file load ##############
address='/home/taruna2/fall2021/output_p3lstm2025/test_nonstationary_transfmer/'
plotadd='/home/.../plotfolder/'

staticndviadd_learn='/home/.../static_ndvi/'
slope_r='/home/.../slope_r/'
aspect_r='/home/.../aspect_r/'
porosity_r='/home/.../porosity_r/'
ndvi_r='/home/.../ndvi_r/'
all_r='/home/.../all_r/'



prediction_horizon=[30,40,50,100,150]
history_len=[10]
fr=0
var_listn=['Ca','Cl','Mg','NO3','K','Na','SO4']
var_listn1=['$Ca^{2+}$','$Cl^-$','$Mg^{2+}$','$NO^-_3$','$K^+$','$Na^+$','$SO^{2-}_4$']
len_ph=len(prediction_horizon)
len_hl=len(history_len)
#
rmse,rmse_sl = np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8))
nse,nse_area,nse_sl,nse_sl_area = np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8))
cme,cme_sl=np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8))
kge,kge_sl=np.empty ((len_ph, len_hl, 8)),np.empty ((len_ph, len_hl, 8))
for i in range(len(prediction_horizon)):
    for j in range(len(history_len)):
        ph=prediction_horizon[i]
        hl=history_len[j]
        pred,obs,rmse[i,j,:],nse[i,j,:],nse_area[i,j,:],cme[i,j,:],kge[i,j,:]=predobs_rmse_nse_pickle(ph,hl,address,'yes')
        pred, obs, rmse_sl[i, j, :], nse_sl[i, j, :], nse_sl_area[i, j, :],cme_sl[i,j,:],kge_sl[i,j,:] = predobs_rmse_nse_pickle (ph, hl, staticndviadd_learn, 'yes')



n_prediction_horizon=['15 hr','20 hr','1 day','2 days','3 days']
nse_ph_combined1(nse, nse_sl, var_listn1, n_prediction_horizon, plotadd, save='no')


# On all the addresses for the indiviual plots

aspect_add='/home/.../aspect/' ##one by one
slope_add='/home/.../slope/'
porosity_add='/home/.../porosity/'
plotadd='/home/.../plots/'
ndviadd_learn='/home/.../test_ndvionly/'

start=1000
import matplotlib.dates as mdates
t = np.array([datetime(2022,4,1) + timedelta(hours=0.5*i) for i in range(17400)])
t=t.reshape((len(t),1))
t=t[:17400]
t1=t[int(len(t)*.7)+start:int(len(t)*.7)+5000]
obs, pred, temp1, temp2,temp3,temp4,temp5 = predobs_rmse_nse_pickle (150, 10, address, 'yes')  #150,10
Q= obs[start:5000,7]
for i in range(7):
    if i==0:  # which solute
        row_index, col_index = 4,0
        ph = prediction_horizon[row_index]
        hl = history_len[col_index]
        obs1_s_fdr, pred_s_fdr, rmse1, temp2,temp3,temp4,temp5 = predobs_rmse_nse_pickle (ph, hl, staticndviadd_learn, 'yes')

        obs1_fdr, pred_fdr, rmse2, temp2,temp3 ,temp4,temp5= predobs_rmse_nse_pickle (ph, hl, address, 'yes')
        f,ax=plt.subplots()
        ax.plot (t1, obs1_fdr[start:5000, i], label='Observed', color='black')
        ax.plot (t1,pred_fdr[start:5000, i], label='NST', color='darkorange')
        ax.plot(t1,pred_s_fdr[start:5000, i], label='NST-Geo', color='darkorchid')##1f77b4
        t1=np.squeeze(t1)
        ax20 = ax.twinx ()
        ax20.plot (t1,Q, color='#929591')
        ax20.fill_between (np.squeeze(t1), 0, Q, color='#929591', alpha=0.7)
        ax20.set_ylim (0, 300)
        ax20.set_xlim(t1[0],t1[-1])
        ax20.set_ylabel ('$Q$ ($m^3/s$)', fontsize=12)

        vmin, vmax = 0, np.nanmax ([obs1_fdr[start :5000, i], pred_fdr[start :5000, i], pred_s_fdr[start :5000, i]])

        ax.set_ylim(0,vmax+.1*abs(vmax))
        ax.set_title ('{} with ph=3 days, hl=5 hr'.format (var_listn1[i]), fontsize=14)
        ax.set_xlabel ('Date', fontsize=12)
        ax.set_ylabel ('Concentration ($mM$)', fontsize=12)
        myFmt = DateFormatter ("%d-%b")
        ax.xaxis.set_major_formatter (myFmt)
        ax.xaxis.set_major_locator (mdates.WeekdayLocator(interval=2))

        print(rmse1[i])
        print(rmse2[i])
        # plt.savefig(plotadd+ 'pred_obs_' + var_listn[i] + '_ph_' + str(ph) + '_hl_' + str(hl) + '_new.png', dpi=300, bbox_inches='tight')
        plt.show ()




################## paramters check #####################
######### resolution
staticadd_res='/home/.../staticonly/'
ndviadd_res='/home/.../ndvionly/'
plotadd='/home/.../plots/'

prediction_horizon=[150]  #1000,1500
history_len=[10]
var_listn=['Ca','Cl','Mg','NO3','K','Na','SO4']
var_listn1=['$Ca^{2+}$','$Cl^-$','$Mg^{2+}$','$NO^-_3$','$K^+$','$Na^+$','$SO^{2-}_4$']
len_ph=len(prediction_horizon)
len_hl=len(history_len)
res=[67,80,100,134,201,268,402,805]
rmse,rmse_sl = np.empty ((len(res), 8)),np.empty ((len(res), 8))
nse,nse_area,nse_sl,nse_sl_area = np.empty ((len(res), 8)),np.empty ((len(res), 8)),np.empty ((len(res), 8)),np.empty ((len(res), 8))


i,j=0,0
ph,hl=prediction_horizon[i],history_len[j]
for k in range(len(res)):
    pred_st,obs_st,rmse[k,:],nse[k,:],nse_area[k,:]=param_predobs_rmse_nse_pickle(ph,hl,staticadd_res,'yes',res[k])
    pred, obs, rmse_sl[k, :], nse_sl[k, :], nse_sl_area[k, :] = param_predobs_rmse_nse_pickle (ph, hl, ndviadd_res, 'yes',res[k])
#
rmse_sl[4:6,2]=rmse_sl[6,2]

xaxis1=[1200,780,600,390,300,240,150,75]
plt.figure()
plt.plot(xaxis1,rmse[:,2])
plt.plot(xaxis1,rmse_sl[:,2])
plt.legend(['Static','NDVI'], fontsize=12)
plt.xlabel('Resolution ($m$)',fontsize=12)
plt.ylabel('RMSE ($mg/l$)',fontsize=12)
plt.title('RMSE vs Resolution for $Mg^{2+}$',fontsize=14)
plt.gca().invert_xaxis()
plt.show()



#################### static and ndvi parameters
def param_read(prediction_horizon, history_len, staticndviadd_learn,geoid,statickey,ndvikey):
    data=readpickle(staticndviadd_learn, str(int(history_len))+'_predicting_'+str(int(prediction_horizon))+'_max-min_data_flow_dr_.pickle')
    staticparam=[]
    ndviparam=[]
    if geoid==0 or geoid==2:
        stp = data['Static_param']
        for i in range(3):
            stpt=stp[statickey[i]]
            stptw= stpt['mlp.0.weight']
            stptw=stptw.cpu().detach().numpy()
            stptwa=np.abs(stptw)
            staticparam.append(stptwa.mean())
    if geoid==1 or geoid==2:
        ndvip = data['NDVI_param']
        for i in range(14):
            ndvipt=ndvip[ndvikey[i]]
            ndviptw= ndvipt['mlp.0.weight']
            ndviptw=ndviptw.cpu().detach().numpy()
            ndviptwa=np.abs(ndviptw)
            ndviparam.append(ndviptwa.mean())
    if geoid==0:
        return staticparam
    if geoid==1:
        return ndviparam
    if geoid==2:
        return staticparam, ndviparam




geofeatures=['static','ndvi','static-ndvi']
geoid=2

static_param,ndvi_param = np.empty ((len_ph, len_hl, 3)),np.empty ((len_ph, len_hl, 14))
statickey=['aspect', 'slope', 'porosity']
ndvikey=['ndvi230211', 'ndvi230115', 'ndvi221126', 'ndvi221103', 'ndvi221003', 'ndvi220918', 'ndvi220818', 'ndvi220722', 'ndvi220703', 'ndvi220618', \
    'ndvi220523', 'ndvi220507', 'ndvi220403', 'ndvi230315']
ndkey=['Feb', 'Jan', 'lNov', 'eNov', 'Oct', 'Sep', 'Aug', 'lJul', 'eJul', 'Jun', \
    'lMay', 'eMay', 'Apr', 'Mar']
stkey= ['a', 's', 'p']

for i in range(len(prediction_horizon)):
    for j in range(len(history_len)):
        ph=prediction_horizon[i]
        hl=history_len[j]
        if geoid==2:
            static_param[i,j,:],ndvi_param[i,j,:] = param_read(ph, hl, staticndviadd_learn,geoid,statickey,ndvikey)





static_datard1 = readpickle ('/home/.../',
                             'static_data_.pickle')
# static_datard1 = readpickle ('/home/.../',
#                              'normalized_static_data_.pickle')
keys = list (static_datard1.keys ())
values = list (static_datard1.values ())
cleaned_values = list (map (remove_nan_rows_cols, values))
static_datard2 = dict (zip (keys, cleaned_values))
static_datard = coarsen_dict_preserve_nan(static_datard2, factor="scale", method='preserve')

staticndviadd_learn='/home/.../test_staticndvi/'
data=readpickle(staticndviadd_learn, str(10)+'_predicting_'+str(150)+'_max-min_data_flow_dr_.pickle')
static_param=data['Static_param']
ndvi_param=data['NDVI_param']
######static parameter mean across axis =0
static_paramdict = {}
for i in range(3):
    temp=static_param[statickey[i]]
    temp1=temp['mlp.0.weight']
    temp1=temp1.cpu().numpy()
    row_index = np.argmax (np.max (temp1, axis=1))
    meancal = temp1[row_index, :]
    static_datarddim= static_datard[statickey[i]].shape
    meancal = np.resize(meancal, static_datarddim)
    p10 = np.nanpercentile (meancal, 10)
    p90 = np.nanpercentile (meancal, 90)

    # Set values below 10th percentile to 0, above 90th percentile to 1
    meancal = np.where (meancal <= p10, 0.0,
                        np.where (meancal >= p90, 1.0,
                                  (meancal - p10) / (p90 - p10)))
    # wherever the original matrix has nan put nan back to emeancal
    meancal[np.isnan(static_datard[statickey[i]])] = np.nan


    static_paramdict[statickey[i]] = meancal

# ###plot image
for i in range(3):
    if i==0:
        plt.figure()
        plt.imshow(static_paramdict[statickey[i]], cmap='Spectral',interpolation='bilinear')
        plt.title(statickey[i]+' ph=3 days',fontsize=14)
        plt.colorbar()
        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks
        # plt.savefig(plotadd+ statickey[i] + '_ph25_hour_static(ndvi).png', dpi=300, bbox_inches='tight')
        plt.show()

######static parameter mean across axis =0
ndvi_paramdict = {}
for i in range(14):
    temp=ndvi_param[ndvikey[i]]
    temp1=temp['mlp.0.weight']
    temp1=temp1.cpu().numpy()
    row_index = np.argmax (np.max (temp1, axis=1))
    meancal = temp1[row_index, :]
    static_datarddim= static_datard[ndvikey[i]].shape
    meancal = np.resize(meancal, static_datarddim)
    meancal[np.isnan(static_datard[ndvikey[i]])] = np.nan


    ndvi_paramdict[ndvikey[i]] = meancal

##plot image
for i in range(14):
    if i==5:
        plt.figure()
        plt.imshow(ndvi_paramdict[ndvikey[i]], cmap='Greens')
        plt.title(ndkey[i]+' NDVI ph=3 days',fontsize=14)
        plt.colorbar()
        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks
        # plt.savefig(plotadd+ ndvikey[i] + '_ph25_hour_(static)ndvi.png', dpi=300, bbox_inches='tight')
        plt.show()


############### influence of NDVI maps at ph and hl

data=readpickle(staticndviadd_learn, str(10)+'_predicting_'+str(150)+'_max-min_data_flow_dr_.pickle')
static_param=data['Static_param']
ndvi_param=data['NDVI_param']

ndvikey1=['ndvi220403','ndvi220507','ndvi220523','ndvi220618', 'ndvi220703', 'ndvi220722', 'ndvi220818', 'ndvi220918', 'ndvi221003', 'ndvi221103', \
          'ndvi221126', 'ndvi230115', 'ndvi230211', 'ndvi230315']
ndvikey2=['Apr22','May22','lMay22','Jun22','Jul22','lJul22','Aug22','Sep22','Oct22','Nov22','Dec22','Jan23','Feb23','Mar23']
ndvi_paramdict = {}
orndvi_paramdict={}

for i in range(14):
    temp=ndvi_param[ndvikey1[i]]
    temp1=temp['mlp.0.weight']
    temp1=temp1.cpu().numpy()
    row_index = np.argmax (np.mean (temp1, axis=1))
    meancal = temp1[row_index, :]
    # meancal= np.nanmean(temp1, axis=0)
    static_datarddim= static_datard[ndvikey1[i]].shape
    meancal = np.resize(meancal, static_datarddim)
    meancal[np.isnan (static_datard[ndvikey1[i]])] = np.nan
    ndvi_paramdict[ndvikey2[i]] = meancal
    orndvi_paramdict[ndvikey2[i]] = static_datard[ndvikey1[i]]
#
# #### ndvi_paramdict contains different shape arrarys make them uniform

for key in ndvi_paramdict:
    resized_array = np.empty((277,258))
    resized_array[:] = np.nan  # Initialize with NaN
    shapendvi=ndvi_paramdict[key].shape
    resized_array[:shapendvi[0], :shapendvi[1]] = ndvi_paramdict[key] # Resize to 89x85
    ndvi_paramdict[key] = resized_array

    orresized_array = np.empty((277,258))
    orresized_array[:] = np.nan  # Initialize with NaN
    orshapendvi=orndvi_paramdict[key].shape
    orresized_array[:orshapendvi[0], :orshapendvi[1]] = orndvi_paramdict[key] # Resize to 89x85
    orndvi_paramdict[key] = orresized_array


# Create a DataFrame from the ndvi_paramdict
ndvi_data = {key: ndvi_paramdict[key].flatten() for key in ndvi_paramdict}
ndvi_df = pd.DataFrame(ndvi_data)
# Melt the DataFrame for seaborn
ndvi_melted = ndvi_df.melt(var_name='NDVI Parameter', value_name='Parameters')
#
# ########## original data


original_data = {key: orndvi_paramdict[key].flatten() for key in orndvi_paramdict}
original_df = pd.DataFrame(original_data)
original_melted = original_df.melt(var_name='NDVI Parameter', value_name='Original NDVI')


fig, ax1 = plt.subplots(figsize=(14, 6))

# 1) Boxplot for learned parameters (left y-axis)
sns.boxplot(
    x="NDVI Parameter",
    y="Parameters",
    data=ndvi_melted,
    ax=ax1,
    color="blue"
)
ax1.set_ylabel("Learned parameter values", fontsize=16,color='blue')
ax1.tick_params(axis='y', labelcolor="blue")

# 2) Twin axis for original NDVI (right y-axis)
ax2 = ax1.twinx()

# Offset second boxplot positions slightly
positions = np.arange(len(ndvi_paramdict)) + 0.25

sns.boxplot(
    x="NDVI Parameter",
    y="Original NDVI",
    data=original_melted,
    ax=ax2,
    boxprops=dict(alpha=0.4, color="darkorchid"),
    whiskerprops=dict(color="darkorange"),
    capprops=dict(color="darkorchid"),
    medianprops=dict(color="darkorchid"),
)
ax2.set_ylabel("Observed NDVI values", fontsize=16, color="darkorchid")
ax2.tick_params(axis='y', labelcolor="darkorchid")
ax2.set_ylim(.1, 1.5)
# --- Adjust appearance ---
plt.title("Learned NDVI parameters vs observed NDVI maps at ph= 3 days", fontsize=18)
plt.xticks(rotation=45)
ax1.set_xlabel("Date", fontsize=16)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12, rotation=45)
ax1.tick_params(axis='y', labelsize=12)   # for learned parameters (left)
ax2.tick_params(axis='y', labelsize=12)   # for original NDVI (right)
plt.tight_layout()
plt.show()


###Create the box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='NDVI Parameter', y='Parameters', data=ndvi_melted)
plt.title('Box Plot of NDVI Parameters at ph=3 days', fontsize=16)
plt.xticks(rotation=45)
plt.xlabel('NDVI Parameters', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.tight_layout()
# plt.savefig('/home/taruna2/fall2021/output_p3lstm2025/learnt/ndvi_boxplot_ph25_hr.png', dpi=300, bbox_inches='tight')
plt.show()



############################### Observed NDVI across years #############################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# 🔹 STEP 1: Extract NDVI keys
# --------------------------------------------------

ndvi_keys = [k for k in static_datard.keys() if 'ndvi' in k]

# Target years
target_years = ['2017','2022', '2025']

# --------------------------------------------------
# 🔹 STEP 2: Build dataset
# --------------------------------------------------

data_list = []

for key in ndvi_keys:
    # Example key: 'ndvi220722'
    date_str = key.replace('ndvi', '')  # '220722'

    year = '20' + date_str[:2]   # '2022'
    month_num = int(date_str[2:4])  # 07, 08, 09

    # Filter years
    if year not in target_years:
        continue

    # Filter months (Jul, Aug, Sep)
    if month_num not in [7, 8, 9]:
        continue

    month_map = {7: 'Jul', 8: 'Aug', 9: 'Sep'}
    month = month_map[month_num]

    arr = static_datard[key].flatten()
    arr = arr[~np.isnan(arr)]

    data_list.extend([
        {'Month': month, 'Year': year, 'NDVI': val}
        for val in arr
    ])

# --------------------------------------------------
# 🔹 STEP 3: DataFrame
# --------------------------------------------------

ndvi_df = pd.DataFrame(data_list)
ndvi_df = ndvi_df[(ndvi_df['NDVI'] > 0.1) & (ndvi_df['NDVI'] <= 1)]
# Ensure correct ordering
month_order = ['Jul', 'Aug', 'Sep']
year_order = ['2017', '2022', '2025']

ndvi_df['Month'] = pd.Categorical(ndvi_df['Month'], categories=month_order, ordered=True)
ndvi_df['Year'] = pd.Categorical(ndvi_df['Year'], categories=year_order, ordered=True)

# --------------------------------------------------
# 🔹 STEP 4: Plot
# --------------------------------------------------
palette = {
    '2017': 'gray',
    '2022': 'darkorchid',
    '2025': 'steelblue'
}
plt.figure(figsize=(10,6))

sns.boxplot(
    x='Month',
    y='NDVI',
    hue='Year',
    data=ndvi_df,
    showfliers=False,
    palette=palette,
    boxprops=dict(alpha=0.4)
)

plt.ylabel('NDVI', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.title('NDVI Distribution (July–September Across Years)', fontsize=16)

plt.legend(title='Year')
plt.tight_layout()
plt.show()
############################### jenny's question end #############################



####################### static data saved in the monticello_data_2223######################
staticadd='/home/taruna2/fall2021/remote_lstm2022/monticello_data_2223/static-ndvi_data/'
ndvi230211, ndvi230115, ndvi221126, ndvi221103, ndvi221003, ndvi220918, ndvi220818, ndvi220722, ndvi220703, ndvi220618,\
ndvi220523, ndvi220507, ndvi220403, ndvi230315, ndvi170729, ndvi170909, ndvi250708, ndvi250918, ndvi250816, ndvi170703, ndvi180923, ndvi180830, ndvi180717=utils_p3.get_ndvidata()

# ndvi230211t,ndvi230115t, ndvi221126t, ndvi221103t, ndvi221003t, ndvi220918t, ndvi220818t, ndvi220722t, ndvi220703t, ndvi220618t,\
# ndvi220523t, ndvi220507t, ndvi220403t, ndvi230315t=min_max_scaler_nan(ndvi230211),min_max_scaler_nan(ndvi230115), min_max_scaler_nan(ndvi221126), min_max_scaler_nan(ndvi221103), min_max_scaler_nan(ndvi221003),\
# min_max_scaler_nan(ndvi220918), min_max_scaler_nan(ndvi220818), min_max_scaler_nan(ndvi220722), min_max_scaler_nan(ndvi220703), min_max_scaler_nan(ndvi220618),\
# min_max_scaler_nan(ndvi220523), min_max_scaler_nan(ndvi220507), min_max_scaler_nan(ndvi220403), min_max_scaler_nan(ndvi230315)


slope=utils_p3.get_slope()
aspect=utils_p3.get_aspect()
area=utils_p3.get_area()
porosity=utils_p3.get_porosity()
# slopet=min_max_scaler_nan(slope)
# aspectt=min_max_scaler_nan(aspect)
# areat=min_max_scaler_nan(area)
# porosityt=min_max_scaler_nan(porosity)

save_vars = dict ()
save_vars['ndvi230211'] = ndvi230211
save_vars['ndvi230115'] = ndvi230115
save_vars['ndvi221126'] = ndvi221126
save_vars['ndvi221103'] = ndvi221103
save_vars['ndvi221003'] = ndvi221003
save_vars['ndvi220918'] = ndvi220918
save_vars['ndvi220818'] = ndvi220818
save_vars['ndvi220722'] = ndvi220722
save_vars['ndvi220703'] = ndvi220703
save_vars['ndvi220618'] = ndvi220618
save_vars['ndvi220523'] = ndvi220523
save_vars['ndvi220507'] = ndvi220507
save_vars['ndvi220403'] = ndvi220403
save_vars['ndvi230315'] = ndvi230315

save_vars['ndvi170803'] =ndvi170729
save_vars['ndvi170909'] =ndvi170909
save_vars['ndvi250708'] =ndvi250708
save_vars['ndvi250918'] =ndvi250918
save_vars['ndvi250816'] =ndvi250816
save_vars['ndvi170703'] =ndvi170703
save_vars['ndvi180923'] =ndvi180923
save_vars['ndvi180830'] =ndvi180830
save_vars['ndvi180717'] =ndvi180717


save_vars['slope'] = slope
save_vars['aspect'] = aspect
save_vars['area'] = area
save_vars['porosity'] = porosity

#### weather data ######
Weath_add='/home/...weather data monticello/'
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

## aggrigating the data to 1 day frequency
# 2022
df1=pd.DataFrame(Wdata2022[:,1], index=Wdata2022[:,0], columns=['0'])
df_positive1 = df1[df1['0'] >= 0]
df_daily_pptsum22 = df_positive1.resample('30min').sum()

df2=pd.DataFrame(Wdata2022[:,2], index=Wdata2022[:,0], columns=['1'])
df_positive2 = df2[df2['1'] > 0]
df_daily_DDmean22 = df_positive2.resample('30min').mean()

df3=pd.DataFrame(Wdata2022[:,3], index=Wdata2022[:,0], columns=['2'])
df_positive3 = df3[df3['2'] > 0]
df_daily_SRmean22 = df_positive3.resample('30min').mean()

### 2023
df1=pd.DataFrame(Wdata2023[:,1], index=Wdata2023[:,0], columns=['0'])
df_positive1 = df1[df1['0'] >= 0]
df_daily_pptsum23 = df_positive1.resample('30min').sum()

df2=pd.DataFrame(Wdata2023[:,2], index=Wdata2023[:,0], columns=['1'])
df_positive2 = df2[df2['1'] > 0]
df_daily_DDmean23 = df_positive2.resample('30min').mean()

df3=pd.DataFrame(Wdata2023[:,3], index=Wdata2023[:,0], columns=['2'])
df_positive3 = df3[df3['2'] > 0]
df_daily_SRmean23 = df_positive3.resample('30min').mean()

ppt=pd.concat([df_daily_pptsum22[4417:], df_daily_pptsum23], ignore_index=True)
DD=pd.concat([df_daily_DDmean22[4417:], df_daily_DDmean23], ignore_index=True)
SR=pd.concat([df_daily_SRmean22[4417:], df_daily_SRmean23], ignore_index=True)
ppt=ppt[:17400]
DD=DD[:17400]
SR=SR[:17400]

ppttemp=ppt
ppttemp[ppttemp==0]=np.nan
pptt=min_max_scaler_nan(ppttemp)
DDt=min_max_scaler_nan(DD)
SRt=min_max_scaler_nan(SR)

save_vars1 = dict ()
save_vars1['ppt'] = pptt
save_vars1['DD'] = DDt
save_vars1['SR'] = SRt

with open (staticadd + 'static_data_.pickle', 'wb') as handle :
    pickle.dump (save_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open (staticadd + 'normalized_weather_usrb_.5hr_data_.pickle', 'wb') as handle :
    pickle.dump (save_vars1, handle, protocol=pickle.HIGHEST_PROTOCOL)





