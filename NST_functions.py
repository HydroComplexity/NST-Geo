# The Annotated Transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks


# Standard PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pickle
import time
import platform
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
from scipy.ndimage import zoom
from datetime import datetime, timedelta
from data_functions import *
import itertools
from sklearn.preprocessing import LabelEncoder


# My model Functions


class StaticEmbedding (nn.Module) :
    def __init__(self, d_model, input_dim) :
        super ().__init__ ()
        self.mlp = nn.Sequential (
            nn.Linear (input_dim, int (d_model / 2)),
        )

    def forward(self, static_features) :
        # Replace NaNs with zero (or use another imputation strategy if preferred)
        static_features = torch.nan_to_num (static_features, nan=0.0)
        return self.mlp (static_features)  # [B, D]


class StaticEmbedding1 (nn.Module) :
    def __init__(self, d_model, input_dim) :
        super ().__init__ ()
        self.mlp = nn.Sequential (
            nn.Linear (input_dim, d_model),
        )

    def forward(self, static_features) :
        # Replace NaNs with zero (or use another imputation strategy if preferred)
        static_features = torch.nan_to_num (static_features, nan=0.0)
        return self.mlp (static_features)  # [B, D]


class NdviEmbedding (nn.Module) :
    def __init__(self, d_model, input_dim) :
        super ().__init__ ()
        self.mlp = nn.Sequential (
            nn.Linear (input_dim, d_model),
        )

    def forward(self, static_features) :
        # Replace NaNs with zero (or use another imputation strategy if preferred)
        static_features = torch.nan_to_num (static_features, nan=0.0)
        return self.mlp (static_features)  # [B, D]


def staticdata(static_datard) :
    aspt = static_datard['aspect']
    slope = static_datard['slope']
    # area = static_datard['area']
    porosity = static_datard['porosity']

    aspt_data, slope_data, porosity_data = torch.Tensor (aspt), torch.Tensor (slope), torch.Tensor (porosity)  # tarun
    asptimshape, slopeimshape, porosityimshape = aspt_data.size (), slope_data.size (), porosity_data.size ()  # tarun
    asptdim, slopedim, porositydim = asptimshape[0] * asptimshape[1], slopeimshape[0] * slopeimshape[1], \
                                     porosityimshape[0] * porosityimshape[1],  # tarun

    return asptdim, slopedim, porositydim, aspt_data.reshape (-1), slope_data.reshape (-1), porosity_data.reshape (-1)


def ndvidata(static_datard, device) :
    ndvi230211, ndvi230115, ndvi221126, ndvi221103, ndvi221003, ndvi220918, ndvi220818, ndvi220722, ndvi220703, ndvi220618, \
    ndvi220523, ndvi220507, ndvi220403, ndvi230315 = static_datard['ndvi230211'], static_datard['ndvi230115'], \
                                                     static_datard['ndvi221126'], static_datard['ndvi221103'], \
                                                     static_datard['ndvi221003'], static_datard['ndvi220918'], \
                                                     static_datard['ndvi220818'], static_datard['ndvi220722'], \
                                                     static_datard['ndvi220703'], static_datard['ndvi220618'], \
                                                     static_datard['ndvi220523'], static_datard['ndvi220507'], \
                                                     static_datard['ndvi220403'], static_datard['ndvi230315']

    ndvi230211_d, ndvi230115_d, ndvi221126_d, ndvi221103_d, ndvi221003_d, ndvi220918_d, ndvi220818_d, ndvi220722_d, ndvi220703_d, ndvi220618_d, \
    ndvi220523_d, ndvi220507_d, ndvi220403_d, ndvi230315_d = torch.Tensor (ndvi230211), torch.Tensor (
        ndvi230115), torch.Tensor (ndvi221126), \
                                                             torch.Tensor (ndvi221103), torch.Tensor (
        ndvi221003), torch.Tensor (ndvi220918), \
                                                             torch.Tensor (ndvi220818), torch.Tensor (
        ndvi220722), torch.Tensor (ndvi220703), torch.Tensor (ndvi220618), \
                                                             torch.Tensor (ndvi220523), torch.Tensor (
        ndvi220507), torch.Tensor (ndvi220403), torch.Tensor (ndvi230315)
    ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, ndvi220722_s, ndvi220703_s, ndvi220618_s, \
    ndvi220523_s, ndvi220507_s, ndvi220403_s, ndvi230315_s = ndvi230211_d.size (), ndvi230115_d.size (), ndvi221126_d.size (), ndvi221103_d.size (), \
                                                             ndvi221003_d.size (), ndvi220918_d.size (), ndvi220818_d.size (), ndvi220722_d.size (), ndvi220703_d.size (), ndvi220618_d.size (), \
                                                             ndvi220523_d.size (), ndvi220507_d.size (), ndvi220403_d.size (), ndvi230315_d.size ()
    ndvi230211_dim, ndvi230115_dim, ndvi221126_dim, ndvi221103_dim, ndvi221003_dim, ndvi220918_dim, ndvi220818_dim, ndvi220722_dim, ndvi220703_dim, ndvi220618_dim, \
    ndvi220523_dim, ndvi220507_dim, ndvi220403_dim, ndvi230315_dim = ndvi230211_s[0] * ndvi230211_s[1], ndvi230115_s[
        0] * ndvi230115_s[1], ndvi221126_s[0] * ndvi221126_s[1], \
                                                                     ndvi221103_s[0] * ndvi221103_s[1], ndvi221003_s[
                                                                         0] * ndvi221003_s[1], ndvi220918_s[0] * \
                                                                     ndvi220918_s[1], \
                                                                     ndvi220818_s[0] * ndvi220818_s[1], ndvi220722_s[
                                                                         0] * ndvi220722_s[1], ndvi220703_s[0] * \
                                                                     ndvi220703_s[1], \
                                                                     ndvi220618_s[0] * ndvi220618_s[1], ndvi220523_s[
                                                                         0] * ndvi220523_s[1], ndvi220507_s[0] * \
                                                                     ndvi220507_s[1], \
                                                                     ndvi220403_s[0] * ndvi220403_s[1], ndvi230315_s[
                                                                         0] * ndvi230315_s[1]
    del ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, ndvi220722_s, ndvi220703_s, ndvi220618_s, \
        ndvi220523_s, ndvi220507_s, ndvi220403_s, ndvi230315_s
    return ndvi230211_dim, ndvi230115_dim, ndvi221126_dim, ndvi221103_dim, ndvi221003_dim, ndvi220918_dim, ndvi220818_dim, ndvi220722_dim, ndvi220703_dim, ndvi220618_dim, \
           ndvi220523_dim, ndvi220507_dim, ndvi220403_dim, ndvi230315_dim, ndvi230211_d.reshape (-1).to (
        device), ndvi230115_d.reshape (-1).to (device), ndvi221126_d.reshape (-1).to (device), \
           ndvi221103_d.reshape (-1).to (device), ndvi221003_d.reshape (-1).to (device), ndvi220918_d.reshape (-1).to (
        device), ndvi220818_d.reshape (-1).to (device), ndvi220722_d.reshape (-1).to (device), \
           ndvi220703_d.reshape (-1).to (device), ndvi220618_d.reshape (-1).to (device), \
           ndvi220523_d.reshape (-1).to (device), ndvi220507_d.reshape (-1).to (device), ndvi220403_d.reshape (-1).to (
        device), ndvi230315_d.reshape (-1).to (device)


def ndvitimescale(ndvi_embd_out, out_dim, len_ndviim) :
    ndvi_embd_out = ndvi_embd_out.reshape (out_dim, len_ndviim)  # tarun d_model=32
    ndvidate = [np.datetime64 ('2022-04-03'), np.datetime64 ('2022-05-07'), np.datetime64 ('2022-05-23'),
                np.datetime64 ('2022-06-18'), \
                np.datetime64 ('2022-07-03'), np.datetime64 ('2022-07-22'), np.datetime64 ('2022-08-18'),
                np.datetime64 ('2022-09-18'), \
                np.datetime64 ('2022-10-03'), np.datetime64 ('2022-11-03'), np.datetime64 ('2022-11-26'),
                np.datetime64 ('2023-01-15'), \
                np.datetime64 ('2023-02-11'), np.datetime64 ('2023-03-15'), np.datetime64 ('2023-04-15')]

    ndvidate1 = pd.to_datetime (ndvidate)

    ndvi_embd_out1 = torch.empty ((18100, out_dim))
    ct = 0
    delta = timedelta (minutes=30)
    for i in range (len (ndvidate) - 1) :
        timedff = ndvidate1[i + 1] - ndvidate1[i]
        number_of_increments = (timedff.total_seconds ()) / (delta.total_seconds ())
        ndvi_embd_out1[ct :ct + int (number_of_increments), :] = torch.repeat_interleave (
            ndvi_embd_out[:, i].unsqueeze (0), repeats=int (number_of_increments), dim=0)
        ct += int (number_of_increments)

    ndvi_embd_out1 = ndvi_embd_out1[1 :17400, :]  # tarun

    return ndvi_embd_out1


def generate_ndvi_date_strings() :
    """
    Generate array of date strings in format "ndviyymmdd_d" based on NDVI dates
    with 30-minute intervals between consecutive dates.

    Returns:
        list: Array of formatted date strings
    """
    # Original NDVI dates from your code
    ndvidate = [
        np.datetime64 ('2022-04-03'), np.datetime64 ('2022-05-07'), np.datetime64 ('2022-05-23'),
        np.datetime64 ('2022-06-18'), np.datetime64 ('2022-07-03'), np.datetime64 ('2022-07-22'),
        np.datetime64 ('2022-08-18'), np.datetime64 ('2022-09-18'), np.datetime64 ('2022-10-03'),
        np.datetime64 ('2022-11-03'), np.datetime64 ('2022-11-26'), np.datetime64 ('2023-01-15'),
        np.datetime64 ('2023-02-11'), np.datetime64 ('2023-03-15'), np.datetime64 ('2023-04-15')
    ]

    ndvidate1 = pd.to_datetime (ndvidate)
    delta = timedelta (minutes=30)

    date_strings = []
    day_counter = 1  # Counter for the "_d" suffix

    for i in range (len (ndvidate) - 1) :
        timediff = ndvidate1[i + 1] - ndvidate1[i]
        number_of_increments = int (timediff.total_seconds () / delta.total_seconds ())

        # Generate timestamps for this interval
        current_time = ndvidate1[i]
        for j in range (number_of_increments) :
            # Format: ndviyymmdd_d
            formatted_date = f"ndvi{current_time.strftime ('%y%m%d')}"
            date_strings.append (formatted_date)

    # Following your original code logic, return elements [1:17400]
    return date_strings[1 :17400] if len (date_strings) > 17400 else date_strings[1 :]


def NDVIModelContainer(d_model, dimensions, device='cpu') :
    ndvi230211_s = NdviEmbedding (d_model, dimensions['ndvi230211_dim'])
    ndvi230115_s = NdviEmbedding (d_model, dimensions['ndvi230115_dim'])
    ndvi221126_s = NdviEmbedding (d_model, dimensions['ndvi221126_dim'])
    ndvi221103_s = NdviEmbedding (d_model, dimensions['ndvi221103_dim'])
    ndvi221003_s = NdviEmbedding (d_model, dimensions['ndvi221003_dim'])
    ndvi220918_s = NdviEmbedding (d_model, dimensions['ndvi220918_dim'])
    ndvi220818_s = NdviEmbedding (d_model, dimensions['ndvi220818_dim'])
    ndvi220722_s = NdviEmbedding (d_model, dimensions['ndvi220722_dim'])
    ndvi220703_s = NdviEmbedding (d_model, dimensions['ndvi220703_dim'])
    ndvi220618_s = NdviEmbedding (d_model, dimensions['ndvi220618_dim'])
    ndvi220523_s = NdviEmbedding (d_model, dimensions['ndvi220523_dim'])
    ndvi220507_s = NdviEmbedding (d_model, dimensions['ndvi220507_dim'])
    ndvi220403_s = NdviEmbedding (d_model, dimensions['ndvi220403_dim'])
    ndvi230315_s = NdviEmbedding (d_model, dimensions['ndvi230315_dim'])

    return ndvi230315_s, ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, \
           ndvi220722_s, ndvi220703_s, ndvi220618_s, ndvi220523_s, \
           ndvi220507_s, ndvi220403_s


def weatherdata(weather_datard) :
    ppt = weather_datard['ppt'].iloc[:, :].values
    ppt = ppt.astype (np.float32)
    DD = weather_datard['DD'].iloc[:, :].values
    DD = DD.astype (np.float32)
    SR = weather_datard['SR'].iloc[:, :].values
    SR = SR.astype (np.float32)
    ppt[np.isnan (ppt)] = 0
    DD1 = interpolate_nans_1d (DD.reshape ((len (DD),)).copy ())
    DD1 = DD1.reshape ((len (DD), 1))

    wdata = np.hstack ((ppt[:-1, :], DD1[:-1, :]))
    wdata = torch.Tensor (wdata)
    print ('weather data loaded and processes successfully')
    return wdata


def interpolate_nans_1d(arr) :
    nans = np.isnan (arr)
    x = np.arange (len (arr))
    arr[nans] = np.interp (x[nans], x[~nans], arr[~nans])
    return arr


def resize(arr, new_shape) :
    """
    Resize to any shape using scipy.ndimage.zoom
    """
    zoom_factors = (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1])
    return zoom (arr, zoom=zoom_factors, order=1)


def readpickle(address, file_name) :
    with open (address + file_name, 'rb') as f :
        data = pickle.load (f)
    return data


def check_gpu() :
    device = torch.device ('cpu')
    OS = platform.platform ().split ("-")[0]

    # MAC
    # https://developer.apple.com/metal/pytorch/

    if OS == 'macOS' :
        if torch.backends.mps.is_available () :
            device = torch.device ("mps")
        else :
            print ("MPS device not found.")


    # Windows or Linux using cuda
    elif OS == 'Windows' or OS == 'Linux' :
        if torch.cuda.is_available () :
            device = torch.device ("cuda")
        else :
            print ("CUDA device not found.")

    print (OS, device)
    return device


# Need the weights from here for decoder embeddings and converting to required value


# Can add normalization and denormalization in the pre or post process or in the main chunk of code. Need to transfer all the statistics with the data

class NonStationaryEncoderPreProcess (nn.Module) :
    def __init__(self, input_size, d_model) :
        super (NonStationaryEncoderPreProcess, self).__init__ ()
        # Variables
        self.d_model = d_model
        self.input_size = input_size
        # Model Layers

        self.linear = nn.Linear (self.input_size, self.d_model)
        self.relu = nn.ReLU ()

    def forward(self, x) :  # x=src

        mean = torch.mean (x[0], dim=1, keepdim=True)

        std = torch.std (x[0], dim=1, keepdim=True)

        stats = torch.stack ((mean, std))

        out = (x[0] - mean) / std

        out = self.linear (out)

        return out, stats, x[1]  # might not have to use the activation function


class NonStationaryDecoderPreProcess (nn.Module) :
    def __init__(self, input_size, d_model) :
        super (NonStationaryDecoderPreProcess, self).__init__ ()
        # Variables
        self.d_model = d_model
        self.input_size = input_size
        # Model Layers

        self.linear = nn.Linear (self.input_size, self.d_model)
        self.relu = nn.ReLU ()

    def forward(self, x) :
        stats = x[1]
        wdata = x[2]
        x = x[0]

        mean = stats[0]
        std = stats[1]

        out = (x - mean) / std

        out = self.linear (out)

        return out, stats, wdata  # might not have to use the activation function


class PreProcess (nn.Module) :
    def __init__(self, input_size, d_model) :
        super (PreProcess, self).__init__ ()
        # Variables
        self.d_model = d_model
        self.input_size = input_size
        # Model Layers

        self.linear = nn.Linear (self.input_size, self.d_model)
        self.relu = nn.ReLU ()

    def forward(self, x) :
        return self.linear (x)  # might not have to use the activation function


class Projector (nn.Module) :
    def __init__(self, input_length, input_size, d_model, d_ff, d_out, dropout=0.1) :
        super (Projector, self).__init__ ()

        # Variables
        self.input_length = input_length
        self.d_model = d_model
        self.input_size = input_size  # feature length
        self.d_ff = d_ff
        self.dropout = dropout
        self.d_out = d_out

        # Model Layers
        self.linear1 = nn.Linear ((self.input_size * self.input_length) + self.input_size, self.d_model)
        self.linear2 = nn.Linear (self.d_model, self.d_ff)
        self.linear3 = nn.Linear (self.d_ff, self.d_model // 2)
        self.linear4 = nn.Linear (self.d_model // 2, self.d_out)
        self.relu = nn.ReLU ()

        ## Need MLP layers #What size?

    def forward(self, x) :
        # assuming x has stats
        stats = x[1]
        x = x[0]

        temp_x = torch.flatten (x, start_dim=1)
        temp_stats = torch.flatten (stats, start_dim=1)

        inputs = torch.cat ((temp_x, temp_stats), dim=1)

        out = self.relu (self.linear1 (inputs))
        out = self.relu (self.linear2 (out))
        out = self.relu (self.linear3 (out))
        out = self.linear4 (out)

        return out  # might not have to use the activation function


class NonStationaryPostProcess (nn.Module) :
    def __init__(self, d_model, output_size) :
        super (NonStationaryPostProcess, self).__init__ ()
        # Variables
        self.output_size = output_size
        self.d_model = d_model
        # Model Layers

        self.linear = nn.Linear (self.d_model, self.output_size)
        self.relu = nn.ReLU ()

    def forward(self, x, stats) :
        out = self.linear (x)
        # restore statistics here

        mean = stats[0]
        std = stats[1]

        out = (out * std) + mean

        return out  # might not have to use the activation function


class PostProcess (nn.Module) :
    def __init__(self, d_model, output_size) :
        super (PostProcess, self).__init__ ()
        # Variables
        self.output_size = output_size
        self.d_model = d_model
        # Model Layers

        self.linear = nn.Linear (self.d_model, self.output_size)
        self.relu = nn.ReLU ()

    def forward(self, x) :
        return self.linear (x)  # might not have to use the activation function



class PositionalEncoding (nn.Module) :
    "Implement the PE function."

    def __init__(self, static_modules, static_data, ndvi_modules, ndvi_data, staticid, ndviid, weatherid, slopeid,
                 aspectid, porosityid, le, ct, d_model, dropout, seq_length=10) :

        super (PositionalEncoding, self).__init__ ()
        self.dropout = nn.Dropout (p=dropout)
        self.d_model = d_model

        self.staticid = staticid
        self.ndviid = ndviid
        self.weatherid = weatherid
        self.slopeid = slopeid
        self.aspectid = aspectid
        self.porosityid = porosityid
        self.ct = ct
        self.le = le

        # Store embedding modules
        if staticid == 'yes' or staticid == 'no' or staticid == 'random' :
            self.static_embedder = nn.ModuleDict ({
                'aspt_inp' : static_modules['aspt_inp'],
                'slope_inp' : static_modules['slope_inp'],
                'porosity_inp' : static_modules['porosity_inp']
            })

            # Store data as buffers
            for name, data in static_data.items () :
                self.register_buffer (name, data)

        if ndviid == 'yes' or ndviid == 'random' :
            self.ndvi_embedder = nn.ModuleDict ({
                'ndvi230211_s' : ndvi_modules['ndvi230211_s'],
                'ndvi230115_s' : ndvi_modules['ndvi230115_s'],
                'ndvi221126_s' : ndvi_modules['ndvi221126_s'],
                'ndvi221103_s' : ndvi_modules['ndvi221103_s'],
                'ndvi221003_s' : ndvi_modules['ndvi221003_s'],
                'ndvi220918_s' : ndvi_modules['ndvi220918_s'],
                'ndvi220818_s' : ndvi_modules['ndvi220818_s'],
                'ndvi220722_s' : ndvi_modules['ndvi220722_s'],
                'ndvi220703_s' : ndvi_modules['ndvi220703_s'],
                'ndvi220618_s' : ndvi_modules['ndvi220618_s'],
                'ndvi220523_s' : ndvi_modules['ndvi220523_s'],
                'ndvi220507_s' : ndvi_modules['ndvi220507_s'],
                'ndvi220403_s' : ndvi_modules['ndvi220403_s'],
                'ndvi230315_s' : ndvi_modules['ndvi230315_s']
            })
        self.ndvi_data = ndvi_data

        # Compute the positional encodings once in log space.
        pe = torch.zeros (seq_length, d_model)

        position = torch.arange (0, seq_length).unsqueeze (1)
        div_term = torch.exp (torch.arange (0, d_model, 2) * -(math.log (10000.0) / d_model))
        pe[:, 0 : :2] = torch.sin (position * div_term)
        pe[:, 1 : :2] = torch.cos (position * div_term)

        pe = pe.unsqueeze (0)
        self.register_buffer ('pe', pe)

    def forward(self, x) :
        stats = x[1]
        wdata = x[2]
        date_strtemp = wdata[:, :, 2].cpu ().numpy ()
        original_shape = date_strtemp.shape
        date_strflatten = date_strtemp.flatten ()
        date_strflatten = date_strflatten.astype (int)
        date_str = self.le.inverse_transform (date_strflatten)
        date_str = date_str.reshape (original_shape)
        x = x[0]
        x = x + Variable (self.pe[:, :x.size (1)], requires_grad=True)
        if self.staticid == 'yes' or self.staticid == 'no' or self.staticid == 'random' :
            # Compute static embedding efficiently
            embeddings = []
            embeddings.append (self.static_embedder['aspt_inp'] (self.aspt_data))
            embeddings.append (self.static_embedder['slope_inp'] (self.slope_data))
            embeddings.append (self.static_embedder['porosity_inp'] (self.porosity_data))

            # Concatenate and ensure proper dimensions
            st_embd = torch.cat (embeddings, dim=-1)  # Concatenate along last dimension

            # Ensure st_embd has correct dimensions for broadcasting
            while st_embd.dim () < 3 :
                st_embd = st_embd.unsqueeze (0)

            # Broadcast to match x dimensions
            st_embd = st_embd.expand_as (x)

            x = x + st_embd
            if self.ct == 1 :
                print ('Static embedding used in the NST')


        if self.ndviid == 'yes' or self.ndviid == 'random' :
            # Compute NDVI embeddings efficiently
            global_embeddings = []
            unique_dates = np.unique (date_str)
            for k in unique_dates :
                global_embeddings.append (self.ndvi_embedder[k + '_s'] (self.ndvi_data[k + '_d']))
            globalndvi_embd = torch.cat (global_embeddings, dim=-1)
            if self.weatherid == 'yes' :
                globalndvi_embd = globalndvi_embd.view (len (unique_dates),
                                                        x.shape[2] - 2)  # Reshape to match x dimensions
            else :
                globalndvi_embd = globalndvi_embd.view (len (unique_dates), x.shape[2])

            ndvi_embeddings = []
            for i in range (wdata.shape[0]) :
                for j in range (wdata.shape[1]) :
                    if date_str[i, j] in unique_dates :
                        # Use the global embedding for the date
                        ndvi_embeddings.append (globalndvi_embd[unique_dates.tolist ().index (date_str[i, j])])
                    # ndvi_embeddings.append (self.ndvi_embedder[date_str[i,j]+'_s'] (self.ndvi_data[date_str[i,j]+'_d']))
            ndvi_embd = torch.cat (ndvi_embeddings, dim=-1)
            if self.weatherid == 'yes' :
                ndvi_embd = ndvi_embd.view (wdata.shape[0], wdata.shape[1], x.shape[2] - 2)
                ndvi = torch.cat ((ndvi_embd, wdata[:, :, :2]), dim=-1)  # Concatenate along the last dimension
                x = x + ndvi
            else :
                ndvi_embd = ndvi_embd.view (x.shape[0], x.shape[1], x.shape[2])  # Reshape to match x dimensions
                x = x + ndvi_embd
            if self.ct == 1 :
                if self.ndviid == 'yes' and self.weatherid == 'yes' :
                    print ('both NDVI and Weather embedding used in NST')
                elif self.ndviid == 'yes' and self.weatherid == 'no' :
                    print ('NDVI embedding used in the NST, Weatherid is False ')
                elif self.ndviid == 'no' and self.weatherid == 'yes' :
                    print ('NDVIid is False, Weather embeddings used in the NST')
                else :
                    print ('Both NDVIid and Weatherid are False, no embedding used in the NST')
        else :
            if self.weatherid == 'yes' :
                ndvi_embd = torch.zeros (x.shape[0], x.shape[1], x.shape[2] - 2).to (
                    self.device)  # Create a zero tensor if NDVIid is False
                ndvi = torch.cat ((ndvi_embd, wdata[:, :, :2]), dim=-1)  # Concatenate along the last dimension
                x = x + ndvi

        self.ct = 2

        return self.dropout (x), stats


# Self attention
# Key, Query, Values
# perform self attention
# outputs

def attention(query, key, value, mask=None, delta=None, tau=None, dropout=0.0) :
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size (-1)

    matmul_out = torch.matmul (query, key.transpose (-2, -1))
    # need to rescale here
    if delta is not None and tau is not None :
        tau = tau.unsqueeze (1).unsqueeze (1)
        delta = delta.unsqueeze (1).unsqueeze (1)  # already transposed 1xS

        matmul_out = (matmul_out * tau) + delta

    scores = matmul_out / math.sqrt (d_k)  # transpose of th elast 2 dimensions

    if mask is not None :
        scores = scores.masked_fill (mask == 0, -1e9)

    p_attn = F.softmax (scores, dim=-1)
    # (Dropout described below)
    p_attn = F.dropout (p_attn, p=dropout)

    return torch.matmul (p_attn, value), p_attn


class MultiHeadedAttention (nn.Module) :
    def __init__(self, h, d_model, dropout=0.1) :
        "Take in model size and number of heads."
        super (MultiHeadedAttention, self).__init__ ()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones (nn.Linear (d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None, delta=None, tau=None) :
        "Implements Figure 2"
        if mask is not None :
            # Same mask applied to all h heads.
            mask = mask.unsqueeze (1)

        nbatches = query.size (0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l (x).view (nbatches, -1, self.h, self.d_k).transpose (1, 2) for l, x in
                             zip (self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        if delta is not None and tau is not None :
            x, self.attn = attention (query, key, value, mask=mask, delta=delta, tau=tau,
                                      dropout=self.p)  # For encoder part
        else :
            x, self.attn = attention (query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose (1, 2).contiguous ().view (nbatches, -1, self.h * self.d_k)

        return self.linears[-1] (x)


def subsequent_mask(size) :
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = np.triu (np.ones (attn_shape), k=1).astype ('uint8')
    return torch.from_numpy (subsequent_mask) == 0


# Position wise Feed Forward Network

class PositionwiseFeedForward (nn.Module) :
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1) :
        super (PositionwiseFeedForward, self).__init__ ()

        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear (d_model, d_ff)
        self.w_2 = nn.Linear (d_ff, d_model)
        self.dropout = nn.Dropout (dropout)

    def forward(self, x) :
        return self.w_2 (self.dropout (F.relu (self.w_1 (x))))


# # Residuals and Layer norm

class LayerNorm (nn.Module) :
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6) :
        super (LayerNorm, self).__init__ ()
        self.a_2 = nn.Parameter (torch.ones (features))
        self.b_2 = nn.Parameter (torch.zeros (features))
        self.eps = eps

    def forward(self, x) :
        mean = x.mean (-1, keepdim=True)
        std = x.std (-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Decoder

# Utility functions
def clones(module, N) :
    "Produce N identical layers."
    return nn.ModuleList ([copy.deepcopy (module) for _ in range (N)])


class SublayerConnection (nn.Module) :
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout) :
        super (SublayerConnection, self).__init__ ()
        self.norm = LayerNorm (size)
        self.dropout = nn.Dropout (dropout)

    def forward(self, x, sublayer) :
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout (sublayer (self.norm (x)))


# Make Model

# main function that generates the transformer encoder and decoder

class EncoderDecoder (nn.Module) :
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, PreProcess_e, PreProcess_d, PostProcess, Projector_delta,
                 Projector_tau) :  # , src_embed, tgt_embed
        super (EncoderDecoder, self).__init__ ()
        self.encoder = encoder
        self.decoder = decoder
        self.PreProcess_e = PreProcess_e
        self.PreProcess_d = PreProcess_d
        self.PostProcess = PostProcess
        self.Projector_delta = Projector_delta  # delta is of sequence length
        self.Projector_tau = Projector_tau  # tau is scalar

    #         self.src_embed = src_embed
    #         self.tgt_embed = tgt_embed
    #         self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_wdata, tgt_wdata) :
        "Take in and process masked src and target sequences."

        x, stats = self.encode (src, src_mask, src_wdata)

        return self.decode (x, stats, src_mask, tgt, tgt_mask, tgt_wdata)

    def encode(self, src, src_mask, src_wdata) :
        x, stats = self.PreProcess_e ((src, src_wdata))  # non stationary component

        delta = self.Projector_delta ((src, stats[0]))  # mean addition to input
        log_tau = self.Projector_tau ((src, stats[1]))  # log std added to input
        tau = torch.exp (log_tau)

        # only need to add the scalling for the destationary attention

        return self.encoder (x, src_mask, delta, tau), stats

    def decode(self, memory, stats, src_mask, tgt, tgt_mask, tgt_wdata) :
        tgt, stats = self.PreProcess_d ((tgt, stats, tgt_wdata))

        return self.PostProcess (self.decoder (tgt, memory, src_mask, tgt_mask), stats)


# Encoder and encoder layer

class Encoder (nn.Module) :
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N) :
        super (Encoder, self).__init__ ()
        self.layers = clones (layer, N)
        self.norm = LayerNorm (layer.size)

    def forward(self, x, mask, delta, tau) :
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers :
            x = layer (x, mask, delta, tau)
        return self.norm (x)


class EncoderLayer (nn.Module) :
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout) :
        super (EncoderLayer, self).__init__ ()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones (SublayerConnection (size, dropout), 2)
        self.size = size

    def forward(self, x, mask, delta, tau) :
        "Follow Figure 1 (left) for connections."

        # Attention is calculated here
        x = self.sublayer[0] (x, lambda x : self.self_attn (x, x, x, mask, delta, tau))

        return self.sublayer[1] (x, self.feed_forward)


# Decoder and Decoder layer

class Decoder (nn.Module) :
    "Generic N layer decoder with masking."

    def __init__(self, layer, N) :
        super (Decoder, self).__init__ ()
        self.layers = clones (layer, N)
        self.norm = LayerNorm (layer.size)

    def forward(self, x, memory, src_mask, tgt_mask) :
        for layer in self.layers :
            x = layer (x, memory, src_mask, tgt_mask)
        return self.norm (x)


class DecoderLayer (nn.Module) :
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout) :
        super (DecoderLayer, self).__init__ ()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones (SublayerConnection (size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask) :
        "Follow Figure 1 (right) for connections."
        m = memory
        # Masked multi head Attention is calculated here
        x = self.sublayer[0] (x, lambda x : self.self_attn (x, x, x, tgt_mask))

        # Cross Attention is calculated here
        x = self.sublayer[1] (x, lambda x : self.src_attn (x, m, m, src_mask))

        return self.sublayer[2] (x, self.feed_forward)


def make_non_stationary_model(static_module, static_data, ndvi_modules, ndvi_data, staticid, ndviid, weatherid, slopeid,
                              aspectid, porosityid, le, ct, input_length=10, output_length=1, feature_length=6, N=1,
                              d_model=512, d_ff=2048, h=8, dropout=0.1) :
    "Helper: Construct a model from hyperparameters."

    # c = copy.deepcopy
    attn1, attn2, attn3 = MultiHeadedAttention (h, d_model, dropout), MultiHeadedAttention (h, d_model,
                                                                                            dropout), MultiHeadedAttention (
        h, d_model, dropout)
    ff1, ff2 = PositionwiseFeedForward (d_model, d_ff, dropout), PositionwiseFeedForward (d_model, d_ff, dropout)
    position_src = PositionalEncoding (static_module, static_data, ndvi_modules, ndvi_data, staticid, ndviid, weatherid,
                                       slopeid, aspectid, porosityid, le, ct, d_model, dropout, seq_length=input_length)
    position_tgt = PositionalEncoding (static_module, static_data, ndvi_modules, ndvi_data, staticid, ndviid, weatherid,
                                       slopeid, aspectid, porosityid, le, ct, d_model, dropout,
                                       seq_length=output_length)  # need to change this to adapt to tgt seq of variable length

    # main model the functions inside will make copies to stack the layers
    model = EncoderDecoder (

        Encoder (EncoderLayer (d_model, attn1, ff1, dropout), N),
        Decoder (DecoderLayer (d_model, attn2, attn3, ff2, dropout), N),

        nn.Sequential (NonStationaryEncoderPreProcess (feature_length, d_model), position_src),  # out,state->positional
        nn.Sequential (NonStationaryDecoderPreProcess (feature_length, d_model), position_tgt),
        NonStationaryPostProcess (d_model, feature_length),
        Projector (input_length, feature_length, d_model, d_ff, input_length),  # delta= batch_size x seq_len
        Projector (input_length, feature_length, d_model, d_ff, 1)  # tau= batch_size x 1
    )

    # Do not need embeddings
    #         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    #         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    #     for p in model.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform(p)

    return model


class Dataset () :
    def __init__(self, data, history_length=10, prediction_horizon=1, device=None) :
        self.data = data
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.device = device

    def __len__(self) :  # switched on before
        self.len = len (self.data) - self.history_length - self.prediction_horizon
        return self.len

    def __getitem__(self, i) :
        if i < len (self.data) - self.history_length - self.prediction_horizon :  # self.len:
            src = torch.Tensor (self.data[i : i + self.history_length])

            # provide to decoder
            pad = torch.zeros (1, self.data.shape[1])

            pad = pad.to (self.device)

            tgt = torch.Tensor (self.data[i + self.history_length :
                                          i + self.history_length + self.prediction_horizon - 1])

            tgt = torch.cat ((pad, tgt))

            # compare the outputs
            tgt_y = torch.Tensor (self.data[i + self.history_length :
                                            i + self.history_length + self.prediction_horizon])

            # Add masks here
            src_mask = torch.Tensor (0).to (self.device)

            tgt_mask = subsequent_mask (self.prediction_horizon).to (self.device)

            return [src, tgt, tgt_y, src_mask, tgt_mask]

        else :
            return 0


def train_epoch(model, data_loader, wdata_loader, loss_func, optimizer, device=None) :
    model.train ()

    #     start = time.time()
    Loss = 0
    num_batches = len (data_loader)

    for i, (batch, wdatabatch) in enumerate (zip (data_loader, wdata_loader)) :

        [src, tgt, tgt_y, src_mask, tgt_mask] = batch
        [src_wdata, tgt_wdata] = wdatabatch[:2]

        if src_mask.shape[-1] == 0 :
            src_mask = None
        if tgt_mask.shape[-1] == 0 :
            tgt_mask = None

        # Currently predicting one step ahed so no mask required
        model_out = model (src, tgt, src_mask, tgt_mask, src_wdata, tgt_wdata)

        l = loss_func (model_out, tgt_y)
        if not torch.isnan (l) :
            # for param in model.parameters():
            #     print(param.data)
            optimizer.zero_grad ()
            l.backward ()
            optimizer.step ()
            Loss += l.item ()


    end = time.time ()
    return Loss / num_batches


def test_epoch(model, data_loader, wdata_loader, loss_func, device=None) :
    model.eval ()
    #     start = time.time()
    Loss = 0
    num_batches = len (data_loader)

    with torch.no_grad () :
        for i, (batch, wdatabatch) in enumerate (zip (data_loader, wdata_loader)) :
            [src, tgt, tgt_y, src_mask, tgt_mask] = batch
            [src_wdata, tgt_wdata] = wdatabatch[:2]

            if src_mask.shape[-1] == 0 :
                src_mask = None
            if tgt_mask.shape[-1] == 0 :
                tgt_mask = None

            # Currently predicting one step ahed so no mask required

            model_out = model (src, tgt, src_mask, tgt_mask, src_wdata, tgt_wdata)

            l = loss_func (model_out, tgt_y)
            Loss += l.item ()

    end = time.time ()

    return Loss / num_batches


def make_predictions(model, data_loader, wdata_loader, loss_func, flow_dr, device=None) :
    model.eval ()
    #     start = time.time()
    Loss = 0
    num_batches = len (data_loader)

    predictions = torch.Tensor ().to (device)

    with torch.no_grad () :
        for i, (batch, wdatabatch) in enumerate (zip (data_loader, wdata_loader)) :
            # if i%prediction_horizon == 0:
            [src, tgt, tgt_y, src_mask, tgt_mask] = batch
            [src_wdata, tgt_wdata] = wdatabatch[:2]
            if src_mask.shape[-1] == 0 :
                src_mask = None

            if tgt_mask.shape[-1] == 0 :
                tgt_mask = None

            # need to predict recursively
            # cant give tgt to the model
            # predict one step ahead and append it to the inputs and predict again

            tgt_d = torch.zeros_like (tgt[:, 0, :]).unsqueeze (1)
            tgt_wdata_d = torch.Tensor (tgt_wdata[:, 0, :]).unsqueeze (1)
            # 0 array given to the decoder
            # used to recursively predict
            for m in range (tgt.shape[-2]) :
                tgt_mask_recursion = tgt_mask[:, :m + 1, :m + 1]
                # plt.imshow(tgt_mask_recursion[0])
                model_out = model (src, tgt_d, src_mask, tgt_mask_recursion, src_wdata, tgt_wdata_d)

                last_value = model_out[:, -1, :].unsqueeze (1)
                wdata_last_value = torch.Tensor (tgt_wdata[:, m, :]).unsqueeze (1)
                if flow_dr == 'yes' :
                    flow = tgt_y[:, m, -1]
                    flow = torch.reshape (flow, (-1, 1))
                    last_value[:, :, -1] = flow
                # temp=tgt_wdata_y[:,m,-1]
                # temp=torch.reshape(temp,(-1,1))
                # wdata_last_value[:,:,0]=temp
                tgt_d = torch.cat ((tgt_d, last_value), dim=1)
                tgt_wdata_d = torch.cat ((tgt_wdata_d, wdata_last_value), dim=1)

            predictions = torch.cat ((predictions, last_value), dim=0)

            l = loss_func (last_value, tgt_y[:, -1, :].unsqueeze (1))
            Loss += l.item ()

    end = time.time ()

    return predictions, Loss / num_batches


# split training and testing
# normalize data
# make predictions

def split_data(data, ratio) :
    """
    data: np array
    ratio = value between 0 and 1
    """
    train = data[:int (ratio * len (data))]
    test = data[int (ratio * len (data)) :]
    return train, test


def normalize_data(data, method="max-min") :  # assuming that the last column will be the target variable
    """
    data: 2D np array
    method: "max-min" or "mean"

    returns the normalized 2D array and dict of stats used to normalize
    """
    # data_stats
    data_min = torch.min (data, axis=0).values
    data_range = torch.max (data, axis=0).values - data_min
    data_mean = torch.mean (data, axis=0)
    data_std = torch.std (data, axis=0)

    if method == 'max-min' :

        data_normal = (data - data_min) / (data_range)
        # normalized dataset and stats

    elif method == 'mean' :
        # print(data_mean.cpu(),data_std.cpu())
        data_normal = (data - data_mean) / (data_std)


    elif method == 'both' :
        data_normal_mean = (data - data_mean) / (data_std)

        data_min = torch.min (data_normal_mean, axis=0).values
        data_range = torch.max (data_normal_mean, axis=0).values - data_min

        data_normal = (data_normal_mean - data_min) / (data_range)

    else :
        print ('Method cannot be %s use "max-min" or "mean"' % (method))
        return 0

    return data_normal, {'data_mean' : data_mean, 'data_std' : data_std, 'data_min' : data_min,
                         'data_range' : data_range}


# model parameters

def line_plot_attribute(predictions, truth, attribute_number) :
    plt.plot (range (len (truth)), truth[:, attribute_number], '--', label='Truth', alpha=0.75)
    plt.plot (range (len (predictions)), predictions[:, attribute_number], label='Predictions', alpha=0.75)
    return 0


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


def intp(X) :
    training_set0 = np.empty (shape=X.shape)
    xx = X.astype (np.cfloat)
    for i in range (X.shape[1]) :
        xx1 = xx[:, i]
        nans, yy = nan_helper (xx1)
        xx1[nans] = np.interp (yy (nans), yy (~nans), xx1[~nans])
        training_set0[:, i] = xx1
    return training_set0


def staticmodel(d_model, asptdim, slopedim, porositydim) :
    aspt_inp, slope_inp, porosity_inp = StaticEmbedding (d_model - 10, asptdim), StaticEmbedding (d_model - 10,
                                                                                                  slopedim), StaticEmbedding1 (
        d_model - 22, porositydim)  # to make it 32 11,11,10
    return aspt_inp, slope_inp, porosity_inp


def static_slope_model(d_model, slopedim) :
    slope_inp = StaticEmbedding (d_model - 10, slopedim)  # to make it 32 11,11,10
    return slope_inp


def static_aspt_model(d_model, asptdim) :
    aspt_inp = StaticEmbedding (d_model - 10, asptdim)  # to make it 32 11,11,10
    return aspt_inp


def static_porosity_model(d_model, porositydim) :
    porosity_inp = StaticEmbedding1 (d_model - 22, porositydim)  # to make it 32 11,11,10
    return porosity_inp


def ndvimodel(d_model, ndvi230211_dim, ndvi230115_dim, ndvi221126_dim, ndvi221103_dim, ndvi221003_dim, ndvi220918_dim,
              ndvi220818_dim, ndvi220722_dim, ndvi220703_dim, ndvi220618_dim, \
              ndvi220523_dim, ndvi220507_dim, ndvi220403_dim, ndvi230315_dim) :
    ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, ndvi220722_s, ndvi220703_s, ndvi220618_s, \
    ndvi220523_s, ndvi220507_s, ndvi220403_s, ndvi230315_s = NdviEmbedding (d_model, ndvi230211_dim), NdviEmbedding (
        d_model, ndvi230115_dim), NdviEmbedding (d_model, ndvi221126_dim), NdviEmbedding (d_model,
                                                                                          ndvi221103_dim), NdviEmbedding (
        d_model, ndvi221003_dim), NdviEmbedding (d_model, ndvi220918_dim), NdviEmbedding (d_model,
                                                                                          ndvi220818_dim), NdviEmbedding (
        d_model, ndvi220722_dim), NdviEmbedding (d_model, ndvi220703_dim), NdviEmbedding (d_model,
                                                                                          ndvi220618_dim), NdviEmbedding (
        d_model, ndvi220523_dim), NdviEmbedding (d_model, ndvi220507_dim), NdviEmbedding (d_model,
                                                                                          ndvi220403_dim), NdviEmbedding (
        d_model, ndvi230315_dim)
    return ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, ndvi220722_s, ndvi220703_s, ndvi220618_s, \
           ndvi220523_s, ndvi220507_s, ndvi220403_s, ndvi230315_s


def check_gradients(abc) :
    for name, param in abc.named_parameters () :
        if param.grad is not None :
            print (f"{name} gradient norm: {param.grad.norm ().item ()}")
        else :
            print (f"{name} has no gradient!")


def ndvi_random(ndvi_data) :
    mask_ndvi = ~torch.isnan (ndvi_data)

    # Extract non-NaN values and shuffle them
    values_ndvi = ndvi_data[mask_ndvi]
    shuffled_indices_ndvi = torch.randperm (values_ndvi.numel ())
    shuffled_values_ndvi = values_ndvi[shuffled_indices_ndvi]
    # Put shuffled values back, keeping NaNs in place
    shuffled_arr_ndvi = ndvi_data.clone ()
    shuffled_arr_ndvi[mask_ndvi] = shuffled_values_ndvi
    ndvi_data1 = shuffled_arr_ndvi
    return ndvi_data1


def fit_non_stationary_transformer(data, static_datard, staticid, ndviid, weatherid, slopeid, aspectid, porosityid,
                                   train_fr=.7, feature_length=7, history_length=10, prediction_horizon=2, N=1,
                                   d_model=512, d_ff=2048, h=8, dropout=0.1, epochs=30, batch_size=128, flow_dr=True,
                                   savefig=True, showfig=False, savevars=True, saveloc="", device=None,
                                   normalization='max-min') :
    static_data = {}
    if staticid == 'yes' :
        asptdim, slopedim, porositydim, aspt_data, slope_data, porosity_data = staticdata (static_datard)
        aspt_inp, slope_inp, porosity_inp = staticmodel (d_model, asptdim, slopedim, porositydim)

        static_modules = {
            'aspt_inp' : aspt_inp,
            'slope_inp' : slope_inp,
            'porosity_inp' : porosity_inp
        }
        static_data1 = {
            'aspt_data' : aspt_data,
            'slope_data' : slope_data,
            'porosity_data' : porosity_data
        }

        # Move modules to device
        for module in static_modules.values () :
            module.to (device)
        for key, data_tensor in static_data1.items () :
            static_data[key] = data_tensor.to (device)  # CORRECT
    elif slopeid == 'yes' :
        asptdim, slopedim, porositydim, aspt_data, slope_data, porosity_data = staticdata (static_datard)
        aspt_inp, temps, porosity_inp = staticmodel (d_model, asptdim, slopedim, porositydim)
        print ('randomized slopes are in place')
        mask_slope = ~torch.isnan (slope_data)

        # Extract non-NaN values and shuffle them
        values_slope = slope_data[mask_slope]
        shuffled_indices_slope = torch.randperm (values_slope.numel ())
        shuffled_values_slope = values_slope[shuffled_indices_slope]
        # Put shuffled values back, keeping NaNs in place
        shuffled_arr_slope = slope_data.clone ()
        shuffled_arr_slope[mask_slope] = shuffled_values_slope
        slope_data = shuffled_arr_slope

        slope_inp = static_slope_model (d_model, slopedim)
        static_modules = {
            'aspt_inp' : aspt_inp,
            'slope_inp' : slope_inp,
            'porosity_inp' : porosity_inp
        }
        static_data1 = {
            'aspt_data' : aspt_data,
            'slope_data' : slope_data,
            'porosity_data' : porosity_data
        }
        # Move modules to device
        for module in static_modules.values () :
            module.to (device)
        for key, data_tensor in static_data1.items () :
            static_data[key] = data_tensor.to (device)  # CORRECT
    elif aspectid == 'yes' :
        asptdim, slopedim, porositydim, aspt_data, slope_data, porosity_data = staticdata (static_datard)
        temps, slope_inp, porosity_inp = staticmodel (d_model, asptdim, slopedim, porositydim)
        print ('randomized aspect values are in place')
        mask_aspt = ~torch.isnan (aspt_data)

        # Extract non-NaN values and shuffle them
        values_aspt = aspt_data[mask_aspt]
        shuffled_indices_aspt = torch.randperm (values_aspt.numel ())
        shuffled_values_aspt = values_aspt[shuffled_indices_aspt]
        # Put shuffled values back, keeping NaNs in place
        shuffled_arr_aspt = aspt_data.clone ()
        shuffled_arr_aspt[mask_aspt] = shuffled_values_aspt
        aspt_data = shuffled_arr_aspt

        aspt_inp = static_aspt_model (d_model, asptdim)
        static_modules = {
            'aspt_inp' : aspt_inp,
            'slope_inp' : slope_inp,
            'porosity_inp' : porosity_inp
        }
        static_data1 = {
            'aspt_data' : aspt_data,
            'slope_data' : slope_data,
            'porosity_data' : porosity_data
        }
        # Move modules to device
        for module in static_modules.values () :
            module.to (device)
        for key, data_tensor in static_data1.items () :
            static_data[key] = data_tensor.to (device)  # CORRECT
    elif porosityid == 'yes' :
        asptdim, slopedim, porositydim, aspt_data, slope_data, porosity_data = staticdata (static_datard)
        aspt_inp, slope_inp, temps = staticmodel (d_model, asptdim, slopedim, porositydim)
        print ('randomized porosity values are in place')
        mask_porosity = ~torch.isnan (porosity_data)

        # Extract non-NaN values and shuffle them
        values_porosity = porosity_data[mask_porosity]
        shuffled_indices_porosity = torch.randperm (values_porosity.numel ())
        shuffled_values_porosity = values_porosity[shuffled_indices_porosity]
        # Put shuffled values back, keeping NaNs in place
        shuffled_arr_porosity = porosity_data.clone ()
        shuffled_arr_porosity[mask_porosity] = shuffled_values_porosity
        porosity_data = shuffled_arr_porosity

        porosity_inp = static_porosity_model (d_model, porositydim)
        static_modules = {
            'aspt_inp' : aspt_inp,
            'slope_inp' : slope_inp,
            'porosity_inp' : porosity_inp
        }
        static_data1 = {
            'aspt_data' : aspt_data,
            'slope_data' : slope_data,
            'porosity_data' : porosity_data
        }
        # Move modules to device
        for module in static_modules.values () :
            module.to (device)
        for key, data_tensor in static_data1.items () :
            static_data[key] = data_tensor.to (device)  # CORRECT
    elif staticid == 'random' :
        print ('static data is all randomaized')
        asptdim, slopedim, porositydim, aspt_data, slope_data, porosity_data = staticdata (static_datard)
        aspt_inp, slope_inp, porosity_inp = staticmodel (d_model, asptdim, slopedim, porositydim)

        aspt_data1, slope_data1, porosity_data1 = ndvi_random (aspt_data), ndvi_random (slope_data), ndvi_random (
            porosity_data)

        static_modules = {
            'aspt_inp' : aspt_inp,
            'slope_inp' : slope_inp,
            'porosity_inp' : porosity_inp
        }
        static_data1 = {
            'aspt_data' : aspt_data1,
            'slope_data' : slope_data1,
            'porosity_data' : porosity_data1
        }
        # Move modules to device
        for module in static_modules.values () :
            module.to (device)
        for key, data_tensor in static_data1.items () :
            static_data[key] = data_tensor.to (device)  # CORRECT
    else :
        static_modules = None
        static_data = None

    ########### weather data processing
    weather_datard = readpickle ('/home/.../',
                                 'weather_usrb_.5hr_data_.pickle')

    ########## NDVI data processing
    date_str = generate_ndvi_date_strings ()
    date_str = np.array (date_str)

    le = LabelEncoder ()
    encoded = le.fit_transform (date_str)  # numpy array of ints
    date_str_t = torch.tensor (encoded, dtype=torch.long)
    date_str_t = date_str_t.to (device)  # Move to device  decoded = le.inverse_transform(date_str.numpy())

    wdata1 = weatherdata (weather_datard)
    wdata1 = wdata1.to (device)
    wdata1 = torch.cat ((wdata1, date_str_t.unsqueeze (1)), dim=1)  # Add date string as a new column

    ndvi_data = {}
    if ndviid == 'yes' :
        ndvi230211_dim, ndvi230115_dim, ndvi221126_dim, ndvi221103_dim, ndvi221003_dim, ndvi220918_dim, ndvi220818_dim, ndvi220722_dim, ndvi220703_dim, ndvi220618_dim, \
        ndvi220523_dim, ndvi220507_dim, ndvi220403_dim, ndvi230315_dim, ndvi230211_d, ndvi230115_d, ndvi221126_d, ndvi221103_d, \
        ndvi221003_d, ndvi220918_d, ndvi220818_d, ndvi220722_d, ndvi220703_d, ndvi220618_d, \
        ndvi220523_d, ndvi220507_d, ndvi220403_d, ndvi230315_d = ndvidata (static_datard, device)

        dimensions = {
            'ndvi230211_dim' : ndvi230211_dim, 'ndvi230115_dim' : ndvi230115_dim, 'ndvi221126_dim' : ndvi221126_dim,
            'ndvi221103_dim' : ndvi221103_dim, 'ndvi221003_dim' : ndvi221003_dim, 'ndvi220918_dim' : ndvi220918_dim,
            'ndvi220818_dim' : ndvi220818_dim, 'ndvi220722_dim' : ndvi220722_dim, 'ndvi220703_dim' : ndvi220703_dim,
            'ndvi220618_dim' : ndvi220618_dim, 'ndvi220523_dim' : ndvi220523_dim, 'ndvi220507_dim' : ndvi220507_dim,
            'ndvi220403_dim' : ndvi220403_dim, 'ndvi230315_dim' : ndvi230315_dim
        }

        # Create NDVI data dictionary
        ndvi_data1 = {
            'ndvi230211_d' : ndvi230211_d, 'ndvi230115_d' : ndvi230115_d, 'ndvi221126_d' : ndvi221126_d,
            'ndvi221103_d' : ndvi221103_d, 'ndvi221003_d' : ndvi221003_d, 'ndvi220918_d' : ndvi220918_d,
            'ndvi220818_d' : ndvi220818_d, 'ndvi220722_d' : ndvi220722_d, 'ndvi220703_d' : ndvi220703_d,
            'ndvi220618_d' : ndvi220618_d, 'ndvi220523_d' : ndvi220523_d, 'ndvi220507_d' : ndvi220507_d,
            'ndvi220403_d' : ndvi220403_d, 'ndvi230315_d' : ndvi230315_d
        }

        if weatherid == 'yes' :
            out_dim = d_model - 2
            print ('using both ndvi data and weather data')
        else :
            out_dim = d_model
            print ('using ndvi data not using weather data')

        ndvi230315_s, ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, \
        ndvi220722_s, ndvi220703_s, ndvi220618_s, ndvi220523_s, \
        ndvi220507_s, ndvi220403_s = NDVIModelContainer (out_dim, dimensions, device=device)

        ndvi_modules = {
            'ndvi230315_s' : ndvi230315_s, 'ndvi230211_s' : ndvi230211_s, 'ndvi230115_s' : ndvi230115_s,
            'ndvi221126_s' : ndvi221126_s, 'ndvi221103_s' : ndvi221103_s, 'ndvi221003_s' : ndvi221003_s,
            'ndvi220918_s' : ndvi220918_s, 'ndvi220818_s' : ndvi220818_s, 'ndvi220722_s' : ndvi220722_s,
            'ndvi220703_s' : ndvi220703_s, 'ndvi220618_s' : ndvi220618_s, 'ndvi220523_s' : ndvi220523_s,
            'ndvi220507_s' : ndvi220507_s, 'ndvi220403_s' : ndvi220403_s
        }

        for module in ndvi_modules.values () :
            module.to (device)
        for key, data_tensor in ndvi_data1.items () :
            ndvi_data[key] = data_tensor.to (device)  # CORRECT

    elif ndviid == 'random' :
        print ('using random ndvi data')
        ndvi230211_dim, ndvi230115_dim, ndvi221126_dim, ndvi221103_dim, ndvi221003_dim, ndvi220918_dim, ndvi220818_dim, ndvi220722_dim, ndvi220703_dim, ndvi220618_dim, \
        ndvi220523_dim, ndvi220507_dim, ndvi220403_dim, ndvi230315_dim, ndvi230211_d, ndvi230115_d, ndvi221126_d, ndvi221103_d, \
        ndvi221003_d, ndvi220918_d, ndvi220818_d, ndvi220722_d, ndvi220703_d, ndvi220618_d, \
        ndvi220523_d, ndvi220507_d, ndvi220403_d, ndvi230315_d = ndvidata (static_datard, device)

        dimensions = {
            'ndvi230211_dim' : ndvi230211_dim, 'ndvi230115_dim' : ndvi230115_dim, 'ndvi221126_dim' : ndvi221126_dim,
            'ndvi221103_dim' : ndvi221103_dim, 'ndvi221003_dim' : ndvi221003_dim, 'ndvi220918_dim' : ndvi220918_dim,
            'ndvi220818_dim' : ndvi220818_dim, 'ndvi220722_dim' : ndvi220722_dim, 'ndvi220703_dim' : ndvi220703_dim,
            'ndvi220618_dim' : ndvi220618_dim, 'ndvi220523_dim' : ndvi220523_dim, 'ndvi220507_dim' : ndvi220507_dim,
            'ndvi220403_dim' : ndvi220403_dim, 'ndvi230315_dim' : ndvi230315_dim
        }

        # Create NDVI data dictionary
        ndvi230211_d1, ndvi230115_d1, ndvi221126_d1, ndvi221103_d1, ndvi221003_d1, ndvi220918_d1, ndvi220818_d1, \
        ndvi220722_d1, ndvi220703_d1, ndvi220618_d1, ndvi220523_d1, \
        ndvi220507_d1, ndvi220403_d1, ndvi230315_d1 = ndvi_random (ndvi230211_d), ndvi_random (
            ndvi230115_d), ndvi_random (ndvi221126_d), \
                                                      ndvi_random (ndvi221103_d), ndvi_random (
            ndvi221003_d), ndvi_random (ndvi220918_d), ndvi_random (ndvi220818_d), \
                                                      ndvi_random (ndvi220722_d), ndvi_random (
            ndvi220703_d), ndvi_random (ndvi220618_d), \
                                                      ndvi_random (ndvi220523_d), ndvi_random (
            ndvi220507_d), ndvi_random (ndvi220403_d), ndvi_random (ndvi230315_d)

        ndvi_data1 = {
            'ndvi230211_d' : ndvi230211_d1, 'ndvi230115_d' : ndvi230115_d1, 'ndvi221126_d' : ndvi221126_d1,
            'ndvi221103_d' : ndvi221103_d1, 'ndvi221003_d' : ndvi221003_d1, 'ndvi220918_d' : ndvi220918_d1,
            'ndvi220818_d' : ndvi220818_d1, 'ndvi220722_d' : ndvi220722_d1, 'ndvi220703_d' : ndvi220703_d1,
            'ndvi220618_d' : ndvi220618_d1, 'ndvi220523_d' : ndvi220523_d1, 'ndvi220507_d' : ndvi220507_d1,
            'ndvi220403_d' : ndvi220403_d1, 'ndvi230315_d' : ndvi230315_d1
        }

        if weatherid == 'yes' :
            out_dim = d_model - 2
            print ('using both ndvi data and weather data')
        else :
            out_dim = d_model
            print ('using ndvi data not using weather data')

        ndvi230315_s, ndvi230211_s, ndvi230115_s, ndvi221126_s, ndvi221103_s, ndvi221003_s, ndvi220918_s, ndvi220818_s, \
        ndvi220722_s, ndvi220703_s, ndvi220618_s, ndvi220523_s, \
        ndvi220507_s, ndvi220403_s = NDVIModelContainer (out_dim, dimensions, device=device)

        ndvi_modules = {
            'ndvi230315_s' : ndvi230315_s, 'ndvi230211_s' : ndvi230211_s, 'ndvi230115_s' : ndvi230115_s,
            'ndvi221126_s' : ndvi221126_s, 'ndvi221103_s' : ndvi221103_s, 'ndvi221003_s' : ndvi221003_s,
            'ndvi220918_s' : ndvi220918_s, 'ndvi220818_s' : ndvi220818_s, 'ndvi220722_s' : ndvi220722_s,
            'ndvi220703_s' : ndvi220703_s, 'ndvi220618_s' : ndvi220618_s, 'ndvi220523_s' : ndvi220523_s,
            'ndvi220507_s' : ndvi220507_s, 'ndvi220403_s' : ndvi220403_s
        }

        for module in ndvi_modules.values () :
            module.to (device)
        for key, data_tensor in ndvi_data1.items () :
            ndvi_data[key] = data_tensor.to (device)  # CORRECT

    else :
        ndvi_modules = None
        ndvi_data = None

    ########### end of NDVI data processing

    if flow_dr == 'yes' :
        print ('all solutes are drived by observed flow rate')
    else :
        print ('flow rate is not used to drive solute concentrations')

    column_names = data.columns
    feature_length = data.shape[1]

    data = torch.from_numpy (data.values).float ()

    if device != None :
        data = data.to (device)

    train, test = split_data (data, train_fr)
    train_wdata, test_wdata = split_data (wdata1, train_fr)  # NDVI data

    # Normalization will be done inside the model in the preprocessing step

    # st_embd=st_embd.to(device)  # tarun

    train_dataset = Dataset (train, history_length=history_length,
                             prediction_horizon=prediction_horizon, device=device)

    test_dataset = Dataset (test, history_length=history_length,
                            prediction_horizon=prediction_horizon, device=device)

    train_dataset_wdata = Dataset (train_wdata, history_length=history_length,
                                   prediction_horizon=prediction_horizon, device=device)
    test_dataset_wdata = Dataset (test_wdata, history_length=history_length,
                                  prediction_horizon=prediction_horizon, device=device)

    data_dataset = Dataset (data, history_length=history_length,
                            prediction_horizon=prediction_horizon, device=device)

    wdata_dataset = Dataset (wdata1, history_length=history_length,
                             prediction_horizon=prediction_horizon, device=device)

    sampler = RandomSampler (train_dataset)

    # train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
    train_loader = DataLoader (train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader (test_dataset, batch_size=batch_size, shuffle=False)

    data_loader = DataLoader (data_dataset, batch_size=batch_size, shuffle=False)
    wdata_loader = DataLoader (wdata_dataset, batch_size=batch_size, shuffle=False)

    train_wdata_loader = DataLoader (train_dataset_wdata, batch_size=batch_size, sampler=sampler)
    test_wdata_loader = DataLoader (test_dataset_wdata, batch_size=batch_size, shuffle=False)

    ct = 1
    model = make_non_stationary_model (static_modules, static_data, ndvi_modules, ndvi_data, staticid, ndviid,
                                       weatherid, slopeid, aspectid, porosityid, le, ct, input_length=history_length,
                                       output_length=prediction_horizon,
                                       feature_length=feature_length,
                                       N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)

    model = model.to (device)

    loss_func = nn.MSELoss ()

    if staticid == 'yes' or staticid == 'random' :
        if ndviid == 'yes' or ndviid == 'random' :
            optimizer = torch.optim.Adam (
                itertools.chain (model.parameters (), aspt_inp.parameters (), slope_inp.parameters (),
                                 porosity_inp.parameters (), ndvi230115_s.parameters (), ndvi230211_s.parameters (),
                                 ndvi230115_s.parameters (), \
                                 ndvi221126_s.parameters (), ndvi221103_s.parameters (), ndvi221003_s.parameters (),
                                 ndvi220918_s.parameters (), ndvi220818_s.parameters (), \
                                 ndvi220722_s.parameters (), ndvi220703_s.parameters (), ndvi220618_s.parameters (),
                                 ndvi220523_s.parameters (), ndvi220507_s.parameters (), \
                                 ndvi220403_s.parameters ()), lr=0.01)
        else :
            optimizer = torch.optim.Adam (
                itertools.chain (model.parameters (), aspt_inp.parameters (), slope_inp.parameters (),
                                 porosity_inp.parameters ()), lr=0.01)
    else :
        if ndviid == 'yes' or ndviid == 'random' :
            optimizer = torch.optim.Adam (
                itertools.chain (model.parameters (), aspt_inp.parameters (), slope_inp.parameters (),
                                 porosity_inp.parameters (), ndvi230115_s.parameters (), ndvi230211_s.parameters (),
                                 ndvi230115_s.parameters (), \
                                 ndvi221126_s.parameters (), ndvi221103_s.parameters (), ndvi221003_s.parameters (),
                                 ndvi220918_s.parameters (), ndvi220818_s.parameters (), \
                                 ndvi220722_s.parameters (), ndvi220703_s.parameters (), ndvi220618_s.parameters (),
                                 ndvi220523_s.parameters (), ndvi220507_s.parameters (), \
                                 ndvi220403_s.parameters ()), lr=0.01)
        else :
            optimizer = torch.optim.Adam (model.parameters (), lr=0.01)


    train_Losses = []
    test_Losses = []


    print ('prediction horizon:', prediction_horizon)
    print ('history length:', history_length)
    for epoch in trange (epochs) :

        if staticid == 'yes' or staticid == 'no' or staticid == 'random' :
            aspt_inp.train (), slope_inp.train (), porosity_inp.train ()
        if ndviid == 'yes' or weatherid == 'yes' or ndviid == 'random' :
            ndvi230115_s.train (), ndvi230211_s.train (), ndvi221126_s.train (), ndvi221103_s.train (), ndvi221003_s.train (), \
            ndvi220918_s.train (), ndvi220818_s.train (), ndvi220722_s.train (), ndvi220703_s.train (), ndvi220618_s.train (), \
            ndvi220523_s.train (), ndvi220507_s.train (), ndvi220403_s.train (), ndvi230315_s.train ()

        epoch_train_loss = train_epoch (model, train_loader, train_wdata_loader, loss_func, optimizer, device=device)


        if staticid == 'yes' or staticid == 'no' or staticid == 'random' :
            aspt_inp.eval (), slope_inp.eval (), porosity_inp.eval ()

        if ndviid == 'yes' or weatherid == 'yes' or ndviid == 'random' :
            ndvi230115_s.eval (), ndvi230211_s.eval (), ndvi221126_s.eval (), ndvi221103_s.eval (), ndvi221003_s.eval (), \
            ndvi220918_s.eval (), ndvi220818_s.eval (), ndvi220722_s.eval (), ndvi220703_s.eval (), ndvi220618_s.eval (), \
            ndvi220523_s.eval (), ndvi220507_s.eval (), ndvi220403_s.eval (), ndvi230315_s.eval ()

        epoch_test_loss = test_epoch (model, test_loader, test_wdata_loader, loss_func, device=device)
        print ('Epoch {}/{}'.format (epoch + 1, epochs) + ' train loss:', str (epoch_train_loss) + ' test loss:',
               str (epoch_test_loss))

        if epoch > 5 :
            if abs (train_Losses[-1] - epoch_train_loss) < 1e-4 and (epoch_test_loss > (min (test_Losses) * 1.05)) :
                train_Losses.append (epoch_train_loss)
                test_Losses.append (epoch_test_loss)
                break

        train_Losses.append (epoch_train_loss)
        test_Losses.append (epoch_test_loss)

    if staticid == 'yes' or staticid == 'no' or staticid == 'random' :
        aspt_inp.eval (), slope_inp.eval (), porosity_inp.eval ()

    if ndviid == 'yes' or weatherid == 'yes' or ndviid == 'random' :
        ndvi230115_s.eval (), ndvi230211_s.eval (), ndvi221126_s.eval (), ndvi221103_s.eval (), ndvi221003_s.eval (), \
        ndvi220918_s.eval (), ndvi220818_s.eval (), ndvi220722_s.eval (), ndvi220703_s.eval (), ndvi220618_s.eval (), \
        ndvi220523_s.eval (), ndvi220507_s.eval (), ndvi220403_s.eval (), ndvi230315_s.eval ()
    predictions, loss = make_predictions (model, test_loader, test_wdata_loader, loss_func, flow_dr,
                                          device=device)  # test_loader
    predictions_data, loss_data = make_predictions (model, data_loader, wdata_loader, loss_func, flow_dr,
                                                    device=device)  # test_loader

    # only the last element of the sequence as actual prediction
    predictions = predictions[:, -1, :]
    predictions = predictions.reshape (len (predictions), feature_length)
    predictions_data = predictions_data[:, -1, :]
    predictions_data = predictions_data.reshape (len (predictions_data), feature_length)

    # Transfer values back to cpu
    predictions = predictions.cpu ()
    predictions_data = predictions_data.cpu ()

    truth = test[history_length :].cpu ()
    truth_data = data[history_length :].cpu ()

    # Line plots
    plt.figure (figsize=[15, int (math.ceil (feature_length / 2)) * 5])
    for i in range (predictions.shape[-1]) :
        plt.subplot (int (math.ceil (feature_length / 2)), 2, i + 1)
        line_plot_attribute (predictions, truth, i)
        plt.title (column_names[i])
        plt.legend ()
        plt.grid (alpha=0.2)

    plt.suptitle ("Predictions vs. Truth")
    plt.tight_layout ()

    if savefig == True :
        if flow_dr == 'yes' :
            plt.savefig (saveloc + '_flow_dr_prediction_line_flow_dr.jpg', dpi=300)
        else :
            plt.savefig (saveloc + '_prediction_line.jpg', dpi=300)
    if showfig == True :
        plt.show ()

    plt.clf ()

    save_vars = dict ()
    save_vars['Train_loss'] = train_Losses
    save_vars['Test_loss'] = test_Losses
    save_vars['Truth'] = truth
    save_vars['Predictions'] = predictions
    save_vars['Epochs'] = epoch
    save_vars['Model'] = model

    save_vars['Prediction_entire'] = predictions_data
    save_vars['Truth_entire'] = truth_data
    if staticid == 'yes' or staticid == 'no' or staticid == 'random' :
        static_params_dict = {
            'aspect' : aspt_inp.state_dict (),
            'slope' : slope_inp.state_dict (),
            'porosity' : porosity_inp.state_dict ()
        }

        if ndviid == 'no' :
            save_vars['Static_param'] = static_params_dict
        elif ndviid == 'yes' or ndviid == 'random' :
            # Save both static and NDVI parameters
            save_vars['Static_param'] = static_params_dict

            ndvi_params_dict = {
                'ndvi230211' : ndvi230211_s.state_dict (),
                'ndvi230115' : ndvi230115_s.state_dict (),
                'ndvi221126' : ndvi221126_s.state_dict (),
                'ndvi221103' : ndvi221103_s.state_dict (),
                'ndvi221003' : ndvi221003_s.state_dict (),
                'ndvi220918' : ndvi220918_s.state_dict (),
                'ndvi220818' : ndvi220818_s.state_dict (),
                'ndvi220722' : ndvi220722_s.state_dict (),
                'ndvi220703' : ndvi220703_s.state_dict (),
                'ndvi220618' : ndvi220618_s.state_dict (),
                'ndvi220523' : ndvi220523_s.state_dict (),
                'ndvi220507' : ndvi220507_s.state_dict (),
                'ndvi220403' : ndvi220403_s.state_dict (),
                'ndvi230315' : ndvi230315_s.state_dict ()
            }
            save_vars['NDVI_param'] = ndvi_params_dict

    save_vars['Config'] = {
        'staticid' : staticid,
        'ndviid' : ndviid,
        'flow_dr' : flow_dr if 'flow_dr' in locals () else 'no'
    }

    # Save optimizer state if needed
    if 'optimizer' in locals () :
        save_vars['Optimizer'] = optimizer.state_dict ()

    if savevars == True :
        if flow_dr == 'yes' :
            with open (saveloc + '_data_flow_dr_.pickle', 'wb') as handle :
                pickle.dump (save_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else :
            with open (saveloc + '_data_.pickle', 'wb') as handle :
                pickle.dump (save_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, save_vars
