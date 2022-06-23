# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import numpy as np
from SparseImageWarp import sparse_image_warp
import random

def tensor_to_img(spectrogram):
    plt.figure(figsize=(10,6)) 
    plt.imshow(spectrogram[0])
    plt.show()

def time_warp(spec, W, i):

    torch.manual_seed(6+i)
    mel_spec = spec
    spec = spec.view(1, spec.shape[0], spec.shape[1])
    num_rows = spec.shape[2]
    spec_len = spec.shape[1]
    device = spec.device

    # adapted from https://github.com/DemisEom/SpecAugment/ (Apache License 2.0)
    pt = (num_rows - 2* W) * torch.rand([1], dtype=torch.float) + W # random point along the time axis
    src_ctr_pt_freq = torch.arange(0, spec_len // 2)  # control points on freq-axis
    src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
    src_ctr_pts = src_ctr_pts.float().to(device)
    
    # Destination
    w = 2 * W * torch.rand([1], dtype=torch.float) - W

    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
    dest_ctr_pts = dest_ctr_pts.float().to(device)

    # warp
    source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  
    dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  

    if w<0:
        warped_spectro, dense_flows = sparse_image_warp(spec, source_control_point_locations, dest_control_point_locations)
    
    else:
        ##  w>0
        mel_spectrogram_flip = torch.Tensor(torch.flip(mel_spec, [1])).unsqueeze(0)
        w2 = -w
        dest_ctr_pt_time2 = src_ctr_pt_time + w2
        dest_ctr_pts2 = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time2), dim=-1)
        dest_ctr_pts2 = dest_ctr_pts2.float().to(device)
        dest_control_point_locations2 = torch.unsqueeze(dest_ctr_pts2, 0)   

        warped_spectro2, dense_flows = sparse_image_warp(mel_spectrogram_flip, source_control_point_locations, dest_control_point_locations2)
        warped_spectro = torch.flip(warped_spectro2, [2])
          
    return [warped_spectro.squeeze(3), float(pt), float(w)]

def test_time_warp(mel_spectrogram, W, i, if_fig):
    spec, pt, w = time_warp(mel_spectrogram, W, i)

    if if_fig:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(mel_spectrogram)
        ax2.imshow(spec[0])
        plt.show()
    return [spec, pt,w]
        
############   
#Freq
def freq_mask(spec, num_masks=1, replace_with_zero=False, F=3, i=1):
    
    if (spec.shape[0]!=1):
        spec = spec.view(1, spec.shape[0], spec.shape[1])
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    ff = np.zeros([3,2])
    for j in range(0, num_masks):    
        random.seed(6*i+j)
        f = random.randrange(1, F+1)
        
        
        f_zero = random.randrange(1, num_mel_channels - f)

        mask_end = f_zero + f 
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()
        ff[j] = [f_zero, f]
  
    return [cloned, ff]

def test_freq_mask(mel_spectrogram, num_masks, replace_with_zero, F, i, if_fig = 1):
    spec, ff = freq_mask(mel_spectrogram, num_masks, replace_with_zero, F, i)
    if if_fig:
        tensor_to_img(spec)

    return [spec, ff]

###############    
#Time
def time_mask(spec, num_masks=1, replace_with_zero=False, T=3, i = 1):

    if (spec.shape[0]!=1):
        spec = spec.view(1, spec.shape[0], spec.shape[1])

    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    tt = np.zeros([3,2])
    
    for j in range(0, num_masks):
        random.seed(6*i+j)

        t = random.randrange(1, T+1)
        t_zero = random.randrange(1, len_spectro - t)

        mask_end = t_zero + t 
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
        tt[j] = [t_zero, t]
    return [cloned, tt]

def test_time_mask(mel_spectrogram, num_masks, replace_with_zero, T, i, if_fig):
    spec, tt = time_mask(mel_spectrogram, num_masks, replace_with_zero, T, i)
    if if_fig:
        tensor_to_img(spec)
    return [spec, tt]