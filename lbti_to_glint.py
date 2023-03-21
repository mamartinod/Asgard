#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:13:02 2022

@author: mam
"""

import numpy as np
from astropy.io import fits
import os
import h5py

def extract_data(f):
    hdu = fits.open(f)
    data = hdu[1].data['FLX_TOT']
    piston_avg = hdu[1].data['PCPHMEAN']
    piston_rms = hdu[1].data['PCPHSTD']
    piston_cos = hdu[1].data['PCPHMCOS']
    piston_sin = hdu[1].data['PCPHMSIN']
    
    return data, piston_avg, piston_rms, piston_cos, piston_sin

path = '/mnt/96980F95980F72D3/lbti/2015-02-08_APR/'
wavelength = np.array([11*1e3])
beams_couple = {'null1': 'Beams 1/2', 'null2': 'Beams 2/3', 'null3': 'Beams 1/4',
                'null4': 'Beams 3/4', 'null5': 'Beams 3/1', 'null6': 'Beams 4/2'}

null_data = []
photo1 = []
photo2 = []
dark = []
piston_rms = []
piston_avg = []

keyword = 'NULL'
null_list = [path+f for f in os.listdir(path) if keyword in f]
keyword = 'BCKG'
bckg_list = [path+f for f in os.listdir(path) if keyword in f]
keyword = 'PHOT1'
photo1_list = [path+f for f in os.listdir(path) if keyword in f]
keyword = 'PHOT2'
photo2_list = [path+f for f in os.listdir(path) if keyword in f]


obsid_photo1 = np.array([float(os.path.basename(elt)[15:18]) for elt in photo1_list])
obsid_photo2 = np.array([float(os.path.basename(elt)[15:18]) for elt in photo2_list])
obsid_bckg = np.array([float(os.path.basename(elt)[15:18]) for elt in bckg_list])
obsid_null = np.array([float(os.path.basename(elt)[15:18]) for elt in null_list])

idx_photo1 = np.argsort(obsid_photo1)
idx_photo2 = np.argsort(obsid_photo2)
idx_bckg = np.argsort(obsid_bckg)
idx_null = np.argsort(obsid_null)

for i in range(len(null_list)):
    idxn = idx_null[i]
    idxp1 = idx_photo1[i]
    idxp2 = idx_photo2[i]

    photo1 = extract_data(photo1_list[idxp1])[0][:,2]
    photo2 = extract_data(photo2_list[idxp2])[0][:,2]
    null_data, piston_avg, piston_rms, piston_cos, piston_sin = extract_data(null_list[idxn])
    null_data = null_data[:,2]
    piston_avg *= 2.2/11.
    piston_rms *= 2.2/11.
    
    dictio = {'p1': photo1,
              'p2': photo2,
              'p3': np.zeros_like(photo1),
              'p4': np.zeros_like(photo1),
              'Iminus1': null_data,
              'Iminus2': np.zeros_like(null_data),
              'Iminus3': np.zeros_like(null_data),
              'Iminus4': np.zeros_like(null_data),
              'Iminus5': np.zeros_like(null_data),
              'Iminus6': np.zeros_like(null_data),
              'Iplus1': np.zeros_like(null_data),
              'Iplus2': np.zeros_like(null_data),
              'Iplus3': np.zeros_like(null_data),
              'Iplus4': np.zeros_like(null_data),
              'Iplus5': np.zeros_like(null_data),
              'Iplus6': np.zeros_like(null_data),
              'piston_avg': piston_avg,
              'piston_rms': piston_rms,
              'piston_cos': piston_cos,
              'piston_sin': piston_sin}

    new_file = null_list[idxn][:-4] + 'hdf5'

    with h5py.File(new_file, 'a') as f:
        f.attrs['date'] = '2015-02-08'
        f.attrs['nbimg'] = null_data.size
        f.attrs['array shape'] = 'python ndim : (nb frame, wl channel)'
    
        f.create_dataset('wl_scale', data=wavelength)
        f['wl_scale'].attrs['comment'] = 'wl in nm'
    
        for key in dictio.keys():
            f.create_dataset(key, data=dictio[key])
            try:
                f[key].attrs['comment'] = beams_couple[key]
            except KeyError:
                pass        

dictio_dark = {'p1': np.zeros_like(dark),
          'p2': np.zeros_like(dark),
          'p3': np.zeros_like(dark),
          'p4': np.zeros_like(dark),
          'Iminus1': np.zeros_like(dark),
          'Iminus2': np.zeros_like(dark),
          'Iminus3': np.zeros_like(dark),
          'Iminus4': np.zeros_like(dark),
          'Iminus5': np.zeros_like(dark),
          'Iminus6': np.zeros_like(dark),
          'Iplus1': np.zeros_like(dark),
          'Iplus2': np.zeros_like(dark),
          'Iplus3': np.zeros_like(dark),
          'Iplus4': np.zeros_like(dark),
          'Iplus5': np.zeros_like(dark),
          'Iplus6': np.zeros_like(dark),
          'piston_avg': np.zeros_like(dark),
          'piston_rms': np.zeros_like(dark)}

for i in range(len(bckg_list)):
    new_file = bckg_list[i][:-4] + 'hdf5'
    dark = extract_data(bckg_list[i])[:1]
    dark = dark[0][:,2]
    dictio_dark['Iminus1'] = dark
    with h5py.File(new_file, 'a') as f:
        f.attrs['date'] = '2015-02-08'
        f.attrs['nbimg'] = dark.size
        f.attrs['array shape'] = 'python ndim : (nb frame, wl channel)'
    
        f.create_dataset('wl_scale', data=wavelength)
        f['wl_scale'].attrs['comment'] = 'wl in nm'
    
        for key in dictio_dark.keys():
            f.create_dataset(key, data=dictio_dark[key])
            try:
                f[key].attrs['comment'] = beams_couple[key]
            except KeyError:
                pass        
