# -*- coding: utf-8 -*-

import os

# paths to training (train+valid) and testing set, for audio (wav) and annotations (txt).
# Data available on zenodo (DOI: 10.5281/zenodo.6359955)
path_test1618_txt = os.path.join('data', 'labels_test')
path_train161718_txt = os.path.join('data','labels_train_valid')
path_test1618_wav = os.path.join('data', 'test')
path_train161718_wav = os.path.join('data','train_valid' )
 
balance_types = ['full_rec','balanced', 'valid' ]                                   # types of balancing set, for test (no balancing), train (approx 50-50),  and valid (1:10) 
balance_ratios = [None, [50, 25], [400, 100]]                                       # [min # negative chunks per recording (hasnoisy + random), min # of random chunks per recording] depending on balance type
chunk_length_ms = 500                                                               # 500 ms
chunk_overlap = 150                                                                 # 150 ms
calls_0 = ['t', 'g', 'czapla', 'gh', 'puszczyk']                                    # negative labels, calls to cut
calls_1 = ['d', 'd?', 'k', 'k?', 'kwiczol', 'r','r?', 's', 's?', 'skowronek', 'ni'] # calls of interest
calls_unknown = ['???','??? mysz', '??? high freq']                                 # unknown, not sure if it's sound of interest or not
tolerance = 0.004                                                                   # tolerance, in seconds (if less than that length of label in chunks, then we assume there's no voice there)
valid_set = ['9niski_szum_BUK4_20161025_000604', 'BUK5_20180930_000704', 'BUK4_20161024_223604',
             'BUK4_20171001_020404a', 'BUK1_20160914_011604', 'BUK5_20170910_025605',
             'BUK4_20161013_200104', 'BUK5_20181003_235705']                        # validation set recordings (subset of training set folder)
test_rec_to_cut = ['BUK5_20161101_002104a', 'BUK5_20161101_002104b']                # recordings to cut, to get test set 3.1 from test set 3 
columns_dataframe = ['chunk_ids', 'chunk_start', 'chunk_end', 'has_bird',
                     'chunks_species', 'call_id', 'has_unknown', 'has_noise']       # colums name for dataframe with chunks description in a recording

representation_1d = ['8_classic', '8_classic_plus_MIR'];
representation_2d = ['spektrogram' , 'mel-spektrogram', 'multitaper']

# Parameters for function_representations.

repr_1d_summary = ['min', 'max', 'mean', 'std']  # methods to summarize classic parameters 
summary_1d = [1,1,1,1]                           # which methods from the above are used ([1,1,1,1] - all)

# for spectrogram or mel-spectrogram representation:
sr = 44100        # sampling ratio
n_fft = 512       # the number of FFT points
win_length = 512  # windows length
hop_length = 150  # hop length
window = "hann"   # window type
f_min = 4000      # chosen min frequency
f_max = 9500      # chosen max frequency

# for mel-spectrogram:
n_mels = 60       # number of mel filters

# for multitaper parameters (multi-taper time-frequency reassignment (TFR) spectrograms): https://github.com/melizalab/libtfr:
N = 512 
step = 145
Np = 490 
K = 2             # number of tapers
tm = 1 
flock = 0.1 
tlock = 10