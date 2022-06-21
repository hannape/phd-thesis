# IMPORTS
import joblib
import os
import numpy as np
import librosa
import time
#import libtfr
#from sklearn.preprocessing import MinMaxScaler
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def function_representations_all(path_wav, data_settype, if_scaler, repr_1d_summary, summary_1d, sr, chunk_length_ms, \
                          n_fft, win_length, hop_length, window, f_min, f_max, n_mels, N, step, Np, K, tm, flock, tlock):
   
  '''
  Creating dataframes with different representations of the audio chunks

  Args: 
    data_settype (list): output from data_load function, chosen dataset
    if_scaler (bool): is the Min-Max scaler used for representations or not. Scalers calculated for training set.
    other arguments described in config.py
  Returns:
    file_names_set (list)- recording names in a set
    indices (list) - chunks indices chosen from particular recordings
    info_chunksy (list) - informations about chosen chunks - the same data as before in dataframe, but only for chosen chunks. Columns:
      ['chunk_ids', 'chunk_start', 'chunk_end', 'has_bird', 'chunks_species', 'call_id', 'has_unknown', 'has_noise']
    repr_full1, repr_full2_, repr_full4, repr_full5 (lists) - representations for chosen chunks. Repr sizes:
      repr1, spektrogram: (1 rec) chunks x 63 x 148
      repr2, mel-spektrogram: (1 rec) chunks x 60 x 148
      repr3, multitaper: (1 rec) chunks x 64 x 149
      repr4, 8_classic: (1 rec) chunks x 1-4 x 8
      repr5, 8_classic_plus_MIR: (1 rec) chunks x 1-4 x 39
      
   Four representations, without multitaper repr3 (problems with libraries and python versioning, done in a separate env and script/console. 
                                                   Repr3 commented here)   
  ''' 

  ## FUNCTIONS
 
  def my_spektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler = None): 
    ''' Spectrogram representation '''    
    
    stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window= window)   # short time Fourier transform STFT
    stft1 = librosa.amplitude_to_db(np.abs(stft)**2)                                                    # square of absolute value, convert to decibel scale 
    freqs = librosa.core.fft_frequencies(n_fft=n_fft, sr=sr)                                            # finding frequencies
    x,  = np.where( freqs >= min(freqs[(freqs >= f_min)]))
    j,  = np.where( freqs <= max(freqs[(freqs <= f_max)]))
    stft1 = stft1[min(x):max(j),]                                                                       # stft in chosen range

    if np.shape(stft1)[1]!= 148:
      stft1 = np.pad(stft1, ((0, 0), (0, 148 - np.shape(stft1)[1])), 'constant', constant_values=(-100))
      print("padding to ",np.shape(stft1))
    if (scaler!=None):
      return scaler.transform(stft1)
    else:
      return stft1
 
 
  def my_melspektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = None):
    ''' Mel-spectrogram representation '''   
    
    stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window)
    abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)
    melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin = f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
    repr2_melspec = 0.5 * librosa.amplitude_to_db(melspect, ref=1.0)
    if np.shape(repr2_melspec)[1]!= 148:
      repr2_melspec = np.pad(repr2_melspec, ((0, 0), (0, 148 - np.shape(repr2_melspec)[1])), 'constant', constant_values=(-50))    
    if (scaler!=None):
      return scaler.transform(repr2_melspec)   #repr1_spectro
    else:
      return repr2_melspec
    
  # def my_multitaper(y, N, step, Np, K, tm, flock, tlock, f_min, f_max, scaler = None): 
  #   ''' Multitaper + Time-frequency reassignement spectrogram ''' 
  #  
  #   result5b = libtfr.tfr_spec(y, N = N, step = step, Np = Np, K = 2, tm = tm, flock = flock, tlock = tlock)     
  #   freqs, ind = libtfr.fgrid(sr, N, fpass=(f_min,f_max)) 
  #   repr3_multitaper = librosa.amplitude_to_db(result5b[ind,]); # tylko interesujące nas pasmo, w log
  #   #stft1 = librosa.amplitude_to_db(np.abs(stft)**2) 
  #   if np.shape(repr3_multitaper)[1]!= 149:
  #     repr3_multitaper = np.pad(repr3_multitaper, ((0, 0), (0, 149 - np.shape(repr3_multitaper)[1])), 'constant', constant_values=(-100))    
  #   if (scaler!=None):
  #     return scaler.transform(repr3_multitaper )   
  #   else:
  #     return repr3_multitaper
  
  def FeatureSpectralFlux(X):  # source: https://www.audiocontentanalysis.org/code/audio-features/spectral-flux-2/
    ''' Spectral flux parameter '''     
    # difference spectrum 
    X = np.c_[X[:, 0], X]
    afDeltaX = np.diff(X, 1, axis=1)
    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return  vsf[1:]                 

  def classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler4 = None):
    ''' base for 8 classic parameters '''   
      
    S1 = np.abs(librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window))
    freqs = librosa.core.fft_frequencies(n_fft=n_fft, sr=sr) 
    o,  = np.where(freqs >= min(freqs[(freqs >= f_min)]))
    j,  = np.where(freqs <= max(freqs[(freqs <= f_max)]))
    freqs1 = freqs[min(o):max(j),]
    S = S1[min(o):max(j),] 

    param_0 = np.sum(S, axis=0)                                                         # power of a signal. Other version: librosa.feature.spectral_bandwidth(S=S, p = 1)  
    param_1 = librosa.feature.spectral_centroid(S=S, freq=freqs1)                       # centroid, https://www.mathworks.com/help/audio/ref/spectralcentroid.html
    param_2 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 2), 2)  # 2nd order moment 
    param_3 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 3), 3)  # 3rd order moment
    param_4 = np.power(librosa.feature.spectral_bandwidth(S=S, freq=freqs1, p = 4), 4)  # 4th order moment
    skosnosc = param_3[0] / np.power(param_2[0], 1.5)                                   # skewness, other version of formula: skosnosc2 = skew(S, axis=0), https://www.mathworks.com/help/audio/ref/spectralskewness.html, 
    kurtoza =  param_4[0]/ np.power(param_2[0], 2) - 3                                  # kurtosis, other version of formula: kurtoza2 = kurtosis(S, axis=0)
    plaskosc = librosa.feature.spectral_flatness(S=S)                                   # spectral flatness. gmean(S_squared)/np.mean(S_squared)
    
    return param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, S


  def my_8_classic(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler = None): ## TO DO scalers
    ''' 8 basic classic parameters ''' 
    
    param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, _ = classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler)
    nb_summary = np.sum(summary_1d)
    paramsy = [[[] for _ in range(8)] for _ in range(nb_summary)]
    idx = 0

    for m in range(np.shape(summary_1d)[0]):
      if summary_1d[m]:
        f = getattr(np, repr_1d_summary[m])
        paramsy[idx]=[f(param_0), f(param_1), f(param_2), f(param_3), f(param_4), f(skosnosc), f(kurtoza), f(plaskosc)]
        idx += 1
        
    if (scaler!=None):
        return scaler.transform(paramsy)
    else:
        return paramsy

  def my_8_classic_plus_MIR(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler5 = None): 
    ''' 8 basic classic parameters plus 31 chosen MIR (music information retrieval) parameters'''  
    
    param_0, param_1, param_2, param_3, param_4, skosnosc, kurtoza, plaskosc, S = classic_base(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler4)
    nb_summary = np.sum(summary_1d)
    stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    abs2_stft = (stft.real * stft.real) + (stft.imag * stft.imag)
    melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin=f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
    
    mfccs =librosa.feature.mfcc(S=librosa.power_to_db(melspect), n_mfcc=12)                                       # mel frequency cepstral coefficients 
    mfcc_delta = librosa.feature.delta(mfccs)                                                                     # delta features of MFCC
    zcr = sum(librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=hop_length))              # Zero crossing rate. Can be interpreted as a measure of signal noisiness
    contrast = librosa.feature.spectral_contrast(S=S, hop_length=hop_length, n_fft=n_fft, n_bands=2, fmin=f_min)  # spectral contrast
    rolloff = librosa.feature.spectral_rolloff(S=S, hop_length=hop_length, n_fft=n_fft, roll_percent=0.85)        # spectral rolloff, chosen percent
    rms = librosa.feature.rms(S=S, hop_length=hop_length, frame_length=124)                                       # spectral flux
    spectral_flux = FeatureSpectralFlux(S)

    paramsy = [[[] for _ in range(39)] for _ in range(nb_summary)]
    idx = 0

    for m in range(np.shape(summary_1d)[0]):
      if summary_1d[m]:
        f = getattr(np, repr_1d_summary[m])  # which statistic is chosen, repr_1d_summary = ['min', 'max', 'mean', 'std']
        paramsy_mir = [f(param_0), f(param_1), f(param_2), f(param_3), f(param_4), f(skosnosc), f(kurtoza), f(plaskosc)]
        paramsy_mir.extend(f(mfccs, axis = 1).tolist())
        paramsy_mir.extend(f(mfcc_delta, axis = 1).tolist())
        paramsy_mir.extend([f(zcr)])
        paramsy_mir.extend(f(contrast, axis = 1).tolist())
        paramsy_mir.extend([f(rolloff), f(rms), f(spectral_flux)])
        paramsy[idx]= paramsy_mir
        idx += 1

    if (scaler5!=None):
        return scaler5.transform(paramsy)
    else:
        return paramsy

  file_names_set = data_settype[0]
  indices = data_settype[1]
  result_dataframe = data_settype[2] 
  df_to_np = result_dataframe.to_numpy() 

  repr_full1, repr_full2 = [[] for _ in range(np.shape(file_names_set)[0])],[[] for _ in range(np.shape(file_names_set)[0])] #, repr_full3 ,[[] for _ in range(np.shape(file_names_set)[0])]
  repr_full4, repr_full5 = [[] for _ in range(np.shape(file_names_set)[0])], [[] for _ in range(np.shape(file_names_set)[0])]
  info_chunksy = [[[] for _ in range(8)] for _ in range(np.shape(file_names_set)[0])]  
 
  print("Recordings in a set: ",file_names_set) 

  scaler_path = os.path.join('C:\\','Users','szaro','Desktop','jupyter')
  scaler1 = joblib.load(os.path.join(scaler_path,'scaler','scaler_spektrogram')) if if_scaler==True else None  
  scaler2 = joblib.load(os.path.join(scaler_path,'scaler','scaler_mel_spektrogram')) if if_scaler==True else None  
  #scaler3 = joblib.load(os.path.join(scaler_path,'scaler','scaler_multitaper') if if_scaler==True else None 
  scaler4 = joblib.load(os.path.join(scaler_path,'scaler','scaler_8_classic')) if if_scaler==True else None  
  scaler5 = joblib.load(os.path.join(scaler_path,'scaler','scaler_8_classic_plus_MIR')) if if_scaler==True else None  

  for k in range(0,np.shape(file_names_set)[0]): 

    empty_list = np.array([[] for _ in range(np.shape(indices[k])[0])])
    if not any(df_to_np[k][4]) and not any(df_to_np[k][5]):  
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3]] + [empty_list, empty_list] + [np.take(df_to_np[k][i],indices[k]) for i in [6, 7]]
      print(np.shape(info_chunksy[k][5]), 'no calls at all')
      
    elif not any(df_to_np[k][5]):
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3,4]] + [empty_list] + [np.take(df_to_np[k][i],indices[k]) for i in [6, 7]]                  
      print(np.shape(info_chunksy[k][5]), 'no calls of interest')
      
    else:
      info_chunksy[k] = [np.take(df_to_np[k][i],indices[k]) for i in [0,1,2,3,4,5,6,7]]

    start_time = time.time()
    representation1 = np.empty([np.shape(indices[k])[0], 63, 148])
    representation2 = np.empty([np.shape(indices[k])[0], 60, 148])
    #representation3 = np.empty([np.shape(indices[k])[0], 64, 149])
    representation4 = np.empty([np.shape(indices[k])[0], np.sum(summary_1d), 8])
    representation5 = np.empty([np.shape(indices[k])[0], np.sum(summary_1d), 39])

    # loop for representations, reading single chunks 
    for num, i in enumerate(indices[k]):                        
      
      y, sr = librosa.load(path_wav + '/'+ file_names_set[k], sr=sr, offset=result_dataframe['chunk_start'][k][i]/sr, duration=chunk_length_ms/1000)
          
      # representation 1 - spectrogram ------- 63 x 148 ------ 
      representation1[num] = my_spektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, scaler1)
      # representation 2 - mel spectrogram  ------- 60 x 148 ------  
      representation2[num] = my_melspektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler2)
      # representation 3 - multitaper  ------- 64 x 149 ------
      #representation3[num] = my_multitaper(y, N, step, Np, K, tm, flock, tlock, f_min, f_max, scaler3)
      # representation 4 - 8_classic  ------- 4 x 8 ------
      representation4[num] = my_8_classic(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler4)
      # representation 5 - 8_classic_plus_MIR  ------- 4 x 39 ------
      representation5[num] = my_8_classic_plus_MIR(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, summary_1d, scaler5)

    repr_full1[k] = representation1 
    repr_full2[k] = representation2
    #repr_full3[k] = representation3
    repr_full4[k] = representation4
    repr_full5[k] = representation5
    print(k,'-', file_names_set[k], '- chunks:', np.shape(indices[k])[0], '- time:', time.time()-start_time)

  return [file_names_set, indices, info_chunksy, repr_full1, repr_full2, repr_full4, repr_full5]     # repr_full3
    