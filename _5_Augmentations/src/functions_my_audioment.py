# -*- coding: utf-8 -*-

import random
import numpy as np
import librosa
import warnings
import pandas as pd
import os

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_rms, calculate_desired_noise_rms

def my_melspektrogram(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = None):
  stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window)
  abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)
  melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin = f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
  repr2_melspec = 0.5 * librosa.amplitude_to_db(melspect, ref=1.0)
  if np.shape(repr2_melspec)[1]!= 148:
     repr2_melspec = np.pad(repr2_melspec, ((0, 0), (0, 148 - np.shape(repr2_melspec)[1])), 'constant', constant_values=(-50))    
  if (scaler!=None):
    return scaler.transform(repr2_melspec)
  else:
    return repr2_melspec

# for irregular input shape chunks (last chunk of a recording), without padding
def my_melspektrogram2(y, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = None):
  stft = librosa.stft(np.array(y), n_fft=n_fft, win_length=win_length, hop_length= hop_length, window=window)
  abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)
  melspect = librosa.feature.melspectrogram(y=None, S=abs2_stft, sr=sr, n_mels= n_mels, fmin = f_min, fmax=f_max, hop_length=hop_length, n_fft=n_fft)
  repr2_melspec = 0.5 * librosa.amplitude_to_db(melspect, ref=1.0)

  if (scaler!=None):
    return scaler.transform(repr2_melspec)
    print('scaler')
  else:
    return repr2_melspec


def chunks_dataframe(file_names_list, file_names, indexes, sample_size, info_chunksy, test_chunks_sizes, 
                     last_col, columns_dataframe):
    
    names_list = [file_names_list[i] for i in indexes]
    names_id_list = [file_names.index(names_list[i]) for i in range(0,sample_size)]
    
    chunk_rec_ids = []
    for i in range(0,sample_size):
        chunk_rec_ids.append(indexes[i]-sum(test_chunks_sizes[0:names_id_list[i]]))
        
    chunks_start_sample, chunks_end_sample,has_bird_sample, chunks_species_sample = [],[],[],[]
    call_id_sample,has_unknown_sample,has_noise_sample, diff_pred_sample =[],[],[],[]
    for i in range(0,sample_size):
        chunks_start_sample.append(info_chunksy[names_id_list[i]][1][chunk_rec_ids[i]])
        chunks_end_sample.append(info_chunksy[names_id_list[i]][2][chunk_rec_ids[i]])
        has_bird_sample.append(info_chunksy[names_id_list[i]][3][chunk_rec_ids[i]])
        chunks_species_sample.append(info_chunksy[names_id_list[i]][4][chunk_rec_ids[i]])
        call_id_sample.append(info_chunksy[names_id_list[i]][5][chunk_rec_ids[i]])
        has_unknown_sample.append(info_chunksy[names_id_list[i]][6][chunk_rec_ids[i]])
        has_noise_sample.append(info_chunksy[names_id_list[i]][7][chunk_rec_ids[i]])
        diff_pred_sample.append(last_col[indexes[i]])

    df_final = pd.DataFrame(np.transpose((names_list, names_id_list, indexes, chunk_rec_ids,chunks_start_sample, chunks_end_sample,has_bird_sample,
                      chunks_species_sample,call_id_sample,has_unknown_sample,has_noise_sample,diff_pred_sample)), columns = columns_dataframe )
    return df_final

def saving_for_audacity(n_recs, df_final, file_names):
    # saving the predictions for export in Audacity  
    for i in range(0,n_recs): # 
        df0 = df_final.loc[df_final['rec_id'] == i]
        file_audacity_name = 'data_min_' + str(i) + str(file_names[i]) + '.txt'
        print(file_audacity_name)
        np.savetxt(file_audacity_name, np.transpose([np.array(df0["chunk_start"])/44100,np.array(df0["chunk_end"])/44100, np.array(df0['pred'] )]), delimiter='\t', fmt='%.6f')

def saving_for_excel(columns_dataframe, df, name):
    # saving the predictions for export in Excel 
    s = ','.join(columns_dataframe)
    np.savetxt('data_' + name + '_all.csv', np.transpose([np.array(df["rec_name"]),np.array(df["rec_id"]),np.array(df["chunk_ids"]),
                                                 np.array(df["chunkrec_ids"]), np.array(df["chunk_start"])/44100,
                                                 np.array(df["chunk_end"])/44100, np.array(df["has_bird"]),np.array(df["chunks_species"]),
                                                 np.array(df["call_id"]),np.array(df["has_unknown"]),np.array(df["has_noise"]),
                                                 np.array(df[columns_dataframe[-1]]) ]), delimiter='\t', fmt="%s", header = s)


# Next functions taken and adapted from library Audiomentations https://github.com/iver56/audiomentations (MIT license), with own modifications - saving parameters of augmentations


class PitchShiftMy(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""
    # adapted from https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/pitch_shift.py, 
    # added: return number of semitones

    supports_multichannel = True

    def __init__(self, min_semitones=-4, max_semitones=4, p=0.5):
        super().__init__(p)
        assert min_semitones >= -12
        assert max_semitones <= 12
        assert min_semitones <= max_semitones
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_semitones"] = random.uniform(
                self.min_semitones, self.max_semitones
            )

    def apply(self, samples, sample_rate):
        if samples.ndim == 2:
            pitch_shifted_samples = np.copy(samples)
            for i in range(samples.shape[0]):
                pitch_shifted_samples[i] = librosa.effects.pitch_shift(
                    pitch_shifted_samples[i],
                    sample_rate,
                    n_steps=self.parameters["num_semitones"],
                )
        else:
            pitch_shifted_samples = librosa.effects.pitch_shift(
                samples, sample_rate, n_steps=self.parameters["num_semitones"]
            )
        #print(self.parameters["num_semitones"])
        return [pitch_shifted_samples, self.parameters["num_semitones"]]


class ShiftMy(BaseWaveformTransform):
    """    Shift the samples forwards or backwards, with or without rollover  """
    # adapted from https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/shift.py, 
    # added: return number of places to shift

    supports_multichannel = True

    def __init__(
        self,
        min_fraction=-0.5,
        max_fraction=0.5,
        rollover=True,
        fade=False,
        fade_duration=0.01,
        p=0.5,
    ):
        """
        :param min_fraction: float, fraction of total sound length
        :param max_fraction: float, fraction of total sound length
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param fade: When set to True, there will be a short fade in and/or out at the "stitch"
            (that was the start or the end of the audio before the shift). This can smooth out an
            unwanted abrupt change between two consecutive samples (which sounds like a
            transient/click/pop).
        :param fade_duration: If `fade=True`, then this is the duration of the fade in seconds.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_fraction >= -1
        assert max_fraction <= 1
        assert type(fade_duration) in [int, float] or not fade
        assert fade_duration > 0 or not fade
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover
        self.fade = fade
        self.fade_duration = fade_duration

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["num_places_to_shift"] = int(
                round(
                    random.uniform(self.min_fraction, self.max_fraction)
                    * samples.shape[-1]
                )
            )

    def apply(self, samples, sample_rate):
        num_places_to_shift = self.parameters["num_places_to_shift"]
        shifted_samples = np.roll(samples, num_places_to_shift, axis=-1)

        if not self.rollover:
            if num_places_to_shift > 0:
                shifted_samples[..., :num_places_to_shift] = 0.0
            elif num_places_to_shift < 0:
                shifted_samples[..., num_places_to_shift:] = 0.0

        if self.fade:
            fade_length = int(sample_rate * self.fade_duration)

            fade_in = np.linspace(0, 1, num=fade_length)
            fade_out = np.linspace(1, 0, num=fade_length)

            if num_places_to_shift > 0:

                fade_in_start = num_places_to_shift
                fade_in_end = min(
                    num_places_to_shift + fade_length, shifted_samples.shape[-1]
                )
                fade_in_length = fade_in_end - fade_in_start

                shifted_samples[..., fade_in_start:fade_in_end,] *= fade_in[
                    :fade_in_length
                ]

                if self.rollover:

                    fade_out_start = max(num_places_to_shift - fade_length, 0)
                    fade_out_end = num_places_to_shift
                    fade_out_length = fade_out_end - fade_out_start

                    shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                        -fade_out_length:
                    ]

            elif num_places_to_shift < 0:

                positive_num_places_to_shift = (
                    shifted_samples.shape[-1] + num_places_to_shift
                )

                fade_out_start = max(positive_num_places_to_shift - fade_length, 0)
                fade_out_end = positive_num_places_to_shift
                fade_out_length = fade_out_end - fade_out_start

                shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                    -fade_out_length:
                ]

                if self.rollover:
                    fade_in_start = positive_num_places_to_shift
                    fade_in_end = min(
                        positive_num_places_to_shift + fade_length,
                        shifted_samples.shape[-1],
                    )
                    fade_in_length = fade_in_end - fade_in_start
                    shifted_samples[..., fade_in_start:fade_in_end,] *= fade_in[
                        :fade_in_length
                    ]
        return [shifted_samples, num_places_to_shift]


class AddBackgroundNoiseMy(BaseWaveformTransform):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.
    Note that the gain of the added noise is relative to the amount of signal in the input. This
    implies that if the input is completely silent, no noise will be added.
    Here are some examples of datasets that can be downloaded and used as background noise:
    * https://github.com/karolpiczak/ESC-50#download
    * https://github.com/microsoft/DNS-Challenge/
    """
    # adapted https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/add_background_noise.py
    # added: return parameter SNR in db
    def __init__(
        self,
        #sounds_path=None,
        min_snr_in_db=3,
        max_snr_in_db=30,
        p=0.5,
        lru_cache_size=2,
        i_bg_chunk = 2,
        df_orig=None,

    ):
        """
        :param sounds_path: Path to a folder that contains sound files to randomly mix in. These
            files can be flac, mp3, ogg or wav.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        super().__init__(p)
        #self.sound_file_paths = get_file_paths(sounds_path)
        #self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        #assert len(self.sound_file_paths) > 0
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.parameters["i_bg_chunk"] = i_bg_chunk
        self.parameters["df_orig"] = df_orig
        # self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(
        #     AddBackgroundNoiseMy._load_sound
        # )

    @staticmethod
    def _load_sound(i_bg, sample_rate, df_orig):
        path_train_wav = os.path.join('..','..','jupyter','data','train_valid' )
        chunk_length_ms = 500
        samples, sr = librosa.load(os.path.join(path_train_wav, df_orig['rec_name'][i_bg]), 
                                    sr =  sample_rate, offset = df_orig['chunk_start'][i_bg]/sample_rate,
                                    duration = chunk_length_ms/1000)
      
        return samples, sr 

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_in_db"] = random.uniform(
                self.min_snr_in_db, self.max_snr_in_db
            )

    def apply(self, samples, sample_rate):
        
        noise_sound, _ = self._load_sound(
            self.parameters["i_bg_chunk"], sample_rate, self.parameters["df_orig"]
        )

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["i_bg_chunk"])
            )
            return samples
        clean_rms = calculate_rms(samples)
        desired_noise_rms = calculate_desired_noise_rms(
            clean_rms, self.parameters["snr_in_db"]
        )

        # Adjust the noise to match the desired noise RMS
        noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        #print(noise_sound)
        return [samples + noise_sound, self.parameters["snr_in_db"]]

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddBackgroundNoise gets discarded when pickling it."
            " E.g. this means the cache will not be used when using AddBackgroundNoise together"
            " with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state
    
class AddGaussianSNRMy(BaseWaveformTransform):
    """
    Add gaussian noise to the samples with random Signal to Noise Ratio (SNR).
    Note that old versions of audiomentations (0.16.0 and below) used parameters
    min_SNR and max_SNR, which had inverse (wrong) characteristics. The use of these
    parameters is discouraged, and one should use min_snr_in_db and max_snr_in_db
    instead now.
    Note also that if you use the new parameters, a random SNR will be picked uniformly
    in the decibel scale instead of a uniform amplitude ratio. This aligns
    with human hearing, which is more logarithmic than linear.
    """
    # adapted https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/add_gaussian_snr.py
    # added: return parameter SNR
    supports_multichannel = True

    def __init__(
        self, min_SNR=None, max_SNR=None, min_snr_in_db=None, max_snr_in_db=None, p=0.5
    ):
        """
        :param min_SNR: Minimum signal-to-noise ratio (legacy). A lower number means less noise.
        :param max_SNR: Maximum signal-to-noise ratio (legacy). A greater number means more noise.
        :param min_snr_in_db: Minimum signal-to-noise ratio in db. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in db. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if min_snr_in_db is None and max_snr_in_db is None:
            # Apply legacy defaults
            if min_SNR is None:
                min_SNR = 0.001
            if max_SNR is None:
                max_SNR = 1.0
        else:
            if min_SNR is not None or max_SNR is not None:
                raise Exception(
                    "Error regarding AddGaussianSNR: Set min_snr_in_db"
                    " and max_snr_in_db to None to keep using min_SNR and"
                    " max_SNR parameters (legacy) instead. We highly recommend to use"
                    " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                    " from legacy parameters to new parameters,"
                    " use the following conversion formulas: \n"
                    "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                    "max_snr_in_db = -20 * math.log10(min_SNR)"
                )
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:

            if self.min_SNR is not None and self.max_SNR is not None:
                if self.min_snr_in_db is not None and self.max_snr_in_db is not None:
                    raise Exception(
                        "Error regarding AddGaussianSNR: Set min_snr_in_db"
                        " and max_snr_in_db to None to keep using min_SNR and"
                        " max_SNR parameters (legacy) instead. We highly recommend to use"
                        " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                        " from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                else:
                    warnings.warn(
                        "You use legacy min_SNR and max_SNR parameters in AddGaussianSNR."
                        " We highly recommend to use min_snr_in_db and max_snr_in_db parameters instead."
                        " To migrate from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                    min_snr = self.min_SNR
                    max_snr = self.max_SNR
                    std = np.std(samples)
                    self.parameters["noise_std"] = random.uniform(
                        min_snr * std, max_snr * std
                    )
            else:
                # Pick snr in decibel scale
                snr = random.uniform(self.min_snr_in_db, self.max_snr_in_db)
                self.parameters["snr"] = snr
                clean_rms = calculate_rms(samples)
                noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

                # In gaussian noise, the RMS gets roughly equal to the std
                self.parameters["noise_std"] = noise_rms

    def apply(self, samples, sample_rate):
        noise = np.random.normal(
            0.0, self.parameters["noise_std"], size=samples.shape
        ).astype(np.float32)
        return [samples + noise, self.parameters["snr"]]