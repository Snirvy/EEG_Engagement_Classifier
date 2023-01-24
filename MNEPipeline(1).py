# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:36:01 2022

@author: scrat
"""
#Don't forget to pip install sklearn, mne and tqdm
#!pip install sklearn
#!pip install mne
#!pip install tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

#Location of csv files
path = "C:\\Users\\scrat\\Desktop\\Matlabfiles\\adaptiveexcel\\"
#Used channels (corresponds to dimensions of dataframe)
ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
#Type of channels. Choose from eeg or meg (For unicorn it is all eeg)
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
#Sampling frequenty of EEG (For unicorn it is 250 Hz)
sampling_freq = 250  # in Hertz

def LoadData(path, ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"], ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'], sampling_freq=250, startfilter = 2, endfilter = 30):
    #Creates the info for MNE
    info = mne.create_info(ch_names= ch_names, ch_types= ch_types, sfreq= sampling_freq)
    #Creates standard 1020 electrode placement
    standard_1020 = mne.channels.make_standard_montage('standard_1020')
    Participantlili = []
    os.chdir(path)
    for a in tqdm(os.listdir(), desc = "Loading data: "):
        raw= pd.read_excel(path + a)
        raw = raw[5000:].transpose()
        raw = sklearn.preprocessing.minmax_scale(raw, axis = 0)
        raw = mne.io.RawArray(raw, info)
        raw = raw.filter(startfilter, endfilter)
        print(len(raw[0][0][0]))
        raw.set_montage(standard_1020)
        #raw.save("EEGDatafiles\\Datafile"+ a[:-5]+".fif")
        Participantlili.append(raw)
    return Participantlili

                
Participantlili= LoadData(path, ch_names, ch_types, sampling_freq)

print(Participantlili[0])


#Excludes bad channels
def ExcludeBads(Participantlili = Participantlili, ch_names =  ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],numberofchannels = 8, nameofnewfolder = "AAARandomCleanFIF"):
    
    count = 0
    #Creates empty folder. AAA is needed to make the folder appear on top.
    if nameofnewfolder not in os.listdir():
        os.mkdir(nameofnewfolder)
    lili = list(os.listdir())
    print(lili)
    for raw in Participantlili:
        count +=1
        for channel in range(numberofchannels):
            if raw[channel][0].mean() == 0.0 and raw[channel][0].max() == 0.0:
                raw.info['bads'] = [ch_names[channel]]
        raw.save(nameofnewfolder +"\\Datafile"+lili[count][:-5]+".fif", overwrite = True)

ExcludeBads(nameofnewfolder = "AAAAdaptiveCleanFIF")


#%%
#raw= pd.read_excel("C:\\Users\\scrat\\Desktop\\Matlabfiles\\test\\raweeg_r10enter.xlsx")
#raw = raw.transpose()
#raw = sklearn.preprocessing.minmax_scale(raw, axis = 0)
#raw = mne.io.RawArray(raw, info)
#raw = raw.filter(2, 30)
#print(len(raw[0][0][0]))
#raw.set_montage(standard_1020)
#print(raw)
raw.save()
#print(os.listdir())
#%%
raw2 = Participantlili [3]
raw2.describe() 

ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
for raw in Participantlili:
    for a in range(8):
        if raw[a][0].mean() == 0.0 and raw[a][0].max() == 0.0:
            raw.info['bads'] = [ch_names[a]]

#print(raw2.info["bads"])
#print(raw2[5][0][0])
#%%
raww = Participantlili[0]
print(raww.info)

rawww = raww.pick_types(eeg=True, exclude=["bads"])
print(rawww)
#%%

print(len(Participantlili))

#%%
for raw in Participantlili:
    rangenumber = len(raw[0][0])// 125
    #Calculate Beta power
    Betalili = []
    for window in range(rangenumber-1):
        rawpsd = raw.compute_psd(tmin =window, tmax = window +1, fmin =12, fmax = 30, picks = "eeg",n_fft = 125)
        psds, freqs = rawpsd.get_data(return_freqs=True)
        Betalili.append(psds)
    
    
    #Calculate Theta power
    Thetalili = []
    for window in range(rangenumber-1):
        rawpsd = raw.compute_psd(tmin =window, tmax = window +1, fmin =4, fmax = 8, picks = "eeg", n_fft = 125)
        psds, freqs = rawpsd.get_data(return_freqs=True)
        Thetalili.append(psds)
    
    #Calculate Alpha power
    Alphalili = []
    for window in range(rangenumber-1):
        rawpsd = raw.compute_psd(tmin =window, tmax = window +1, fmin =8, fmax = 12, picks = "eeg", n_fft = 125)
        psds, freqs = rawpsd.get_data(return_freqs=True)
        Alphalili.append(psds)
    
    Engagementlili = []
    for a in range(len(Alphalili)):
        Engagement = np.sum(Betalili[a][0][:]) / (np.sum(Alphalili[a][0][:]) + np.sum(Thetalili[a][0][:]))
        Engagementlili.append(Engagement)
    print(Engagementlili)

#%%
#print(os.listdir())
#mne.io.read_raw_fif(fname, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None)
#os.chdir("AAAAdaptiveCleanFIF")
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from mne import io
from mne_connectivity import spectral_connectivity_epochs, seed_target_indices

def loadAdaptive():
    
    os.chdir("C:\\Users\\scrat\\Desktop\\Matlabfiles\\adaptiveexcel\\AAAAdaptiveCleanFIF")
    oslist = list(os.listdir())
    Adaptiverawlili = []
    for a in oslist:
        Adaptiverawlili.append(mne.io.read_raw_fif(a, preload = False))
    return Adaptiverawlili

def loadRandom():
    
    os.chdir("C:\\Users\\scrat\\Desktop\\Matlabfiles\\randomexcel\\AAARandomCleanFIF")
    oslist = list(os.listdir())
    Randomrawlili = []
    for a in oslist:
        Randomrawlili.append(mne.io.read_raw_fif(a, preload = False))
    return Randomrawlili


Adaptiverawlili = loadAdaptive()
Randomrawlili = loadRandom()
#print("bababowe: ", Person1, Person2, Person3, Person4)
#%%
import scipy
#!pip install xarray --user
import xarray
import joblib
import matplotlib
#!pip install --user mne-connectivity
#!pip install --user -U https://api.github.com/repos/mne-tools/mne-connectivity/zipball/main
#!python -c "import mne_connectivity"
import sys

import importlib.util
spec=importlib.util.spec_from_file_location("mne_connectivity","C:\\Users\\scrat\\Desktop\\Matlabfiles\\randomexcel\\AAARandomCleanFIF\\mne_connectivity\\mne-connectivity-main\\mne_connectivity\\base.py")

 
# creates a new module based on spec
mne_connectivity = importlib.util.module_from_spec(spec)
#sys.path.append(" C:\\Users\\scrat\\Desktop\\Matlabfiles\\randomexcel\\AAARandomCleanFIF\\mne_connectivity\\mne-connectivity-main\\mne_connectivity")

#import __init__
#import io
#import base
#import conftest
#import effective
#import envelope
#print(mne_connectivity.__version__)

#%%

#Adaptiverawlili[0].plot_psd()
#!pip install mne_connectivity
#!pip install -U https://github.com/mne-tools/mne-connectivity/archive/main.zip
#import importlib  
#mne_connectivity = importlib.import_module("mne_connectivity")
#print(os.listdir())
import mne
#from mne_connectivity import spectral_connectivity_epochs
#import mne_connectivity
#freqs = list(range(1,30))
#print(freqs)

#Adaptiverawlili[0].info
#mne.time_frequency.psd_array_welch(Adaptiverawlili[0], sfreq  =250)
#BaseEpochs = tuple
mne_connectivity.SpectroTemporalConnectivity(Adaptiverawlili[0], freqs, sfreq = 250)

#%%
!pip install xarray
#%%
!pip install mne_connectivity
import mne_connectivity
mne_connectivity.spectral_connectivity_time(Adaptiverawlili[0], freqs, sfreq = 250)


