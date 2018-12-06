#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:52:58 2018

@author: lw604
"""
#######################################################################################
## This script is to run T-permutation test for two conditions
## Permute across ROIs: averaged within 300-500ms time window
#######################################################################################

import os.path as op
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from mne import SourceEstimate
import mne
plt.ion()


#######################################################################################
def threshold_by_pvalue(stc_label, threshold_value):
    stc_label.data[stc_label.data<threshold_value]=0
    return stc_label

#-------------------------------------------------------
## Function: get the permutation distribution
#-------------------------------------------------------
def get_Tpermutation_distribution(dat_all, nperm, time_start, time_end, labels):
    dat = dat_all[:,:,int(time_start*1000)+100:int(time_end*1000)+100]
    # permuta the data and calculate the largest sumed p values for each permutation across labels
    highest_permutation_values = []
    sign  = np.ones((dat.shape[0],))
    sign[:np.int32(np.round(len(sign)/2.))] =-1
    for index in range(nperm):#to permute the conditions
        print("permutation:" + str(index))
        sign = np.random.permutation(sign)
        dat_perm = sign * np.transpose(dat,[1,2,0])
        stats_perm, pval_perm = ttest_1samp(dat_perm, 0, axis=2)
        stc_perm = SourceEstimate(-np.log10(pval_perm), vertices=[np.arange(10242), np.arange(10242)],
               tmin=time_start, tstep=1/1000., subject='fsaverage')
        means_of_thresholded_labels = []
        for label in labels:
            label_fname  = op.join('/autofs/cluster/kuperberg/nonconMM/MEG/MNE/250Labels', label + '.label')
            label_name = mne.read_label(label_fname, subject='fsaverage', color='r')
            stc_label_perm = stc_perm.in_label(label_name)
            stc_label_perm = threshold_by_pvalue(stc_label_perm, 2)
            means_of_thresholded_labels.append(stc_label_perm.data.mean())
        highest_permutation_values.append(max(means_of_thresholded_labels))
    return highest_permutation_values

#-------------------------------------------------------
## Run the permutation function: LH ROIs
#-------------------------------------------------------
data_path='/autofs/cluster/kuperberg/nonconMM/MEG/'

def get_Tpermutation_distribution_LH(idx):
    data1 = data1_all[idx]
    data2 = data2_all[idx]
    dat_1 = np.load(op.join(data_path, "MNE", "export",'wholebrain_dSPM_signed_' + data1 + '_keepsubs.npy'))
    dat_2 = np.load(op.join(data_path, "MNE", "export",'wholebrain_dSPM_signed_' + data2 + '_keepsubs.npy'))
    dat_all = dat_1 - dat_2
    for time_start,time_end in time_All:
        highest_permutation_values = get_Tpermutation_distribution(dat_all, nperm, time_start, time_end, labels)
        np.save(op.join(data_path, "MNE", "source_MNE", 'newLH_stat_Tpermutation_Space_dSPM_signed_' + data1 + '_' + data2 +'_' +str(int(time_start*1000)) + '-' + str(int(time_end*1000)) + 'ms.npy'),highest_permutation_values)


#-------------------------------------------------------
## Run the permutation function: RH ROIs
#-------------------------------------------------------
def get_Tpermutation_distribution_RH(idx):
    data1 = data1_all[idx]
    data2 = data2_all[idx]
    dat_1 = np.load(op.join(data_path, "MNE", "export",'wholebrain_dSPM_signed_' + data1 + '_keepsubs.npy'))
    dat_2 = np.load(op.join(data_path, "MNE", "export",'wholebrain_dSPM_signed_' + data2 + '_keepsubs.npy'))
    dat_all = dat_1 - dat_2
    for time_start,time_end in time_All:
        highest_permutation_values = get_Tpermutation_distribution(dat_all, nperm, time_start, time_end, labels)
        np.save(op.join(data_path, "MNE", "source_MNE", 'newRH_stat_Tpermutation_Space_dSPM_signed_' + data1 + '_' + data2 +'_' +str(int(time_start*1000)) + '-' + str(int(time_end*1000)) + 'ms.npy'),highest_permutation_values)

#-------------------------------------------------------
## Run paralell scripts
#-------------------------------------------------------
## define time windows: 300-500ms, 600-1000ms
#time_All = [(0.3,0.5),(0.6,1.0)]
time_All = [(0.3,0.5)]
nperm = 10000 # Nr of permutation
data1_all = ['2245','222','2237','222','2237','2223457','224','223','222','2268','2268','2268','222','223','222','22245']
data2_all = ['221','221','221','2245','2245','221','221','221','2237','221','2245','2237','224','224','223','221']


from joblib import Parallel, delayed
## get the ROI in the left hemisphere
labels_lh_fname = op.join("/autofs/cluster/kuperberg/nonconMM/MEG/MNE/250Labels", "ROI_LH_temp_frontal_parietal.txt")
labels  = [line.strip() for line in open(labels_lh_fname, 'r')]
Parallel(n_jobs=8,verbose=8)(delayed(get_Tpermutation_distribution_LH)(idx) for idx in range(0,len(data1_all)))
