#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:16:40 2022

@author: mam
"""

import numpy as np
from astropy.constants import h, c, k_B
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, quad

def load_lamp(path, power):
    data = np.loadtxt(path)
    wl = data[:,0] * u.nm
    spectrum = data[:,1]
    spectrum = spectrum / np.sum(spectrum * np.mean(np.diff(wl))) * power # We have the spectral power density
    bb = data[:,2]
    bb = bb / np.sum(bb * np.mean(np.diff(wl))) * power
    """
    wl is in nm, spectrum is in W/nm
    """
    return wl, spectrum, bb
    
def convert_Watt_to_photons(wl, spectrum, h, c):
    photon = h * c / wl.to(u.m) # Energy of a single photon in J
    psd = spectrum / photon # PSD in ph/s/nm
    return psd

def plot_lamp(to_plot_wl, to_plot, title, *args):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 6.528 * 1.5
    height = width / 1.618
    sz = 16
    plt.rc('xtick', labelsize=sz)
    plt.rc('ytick', labelsize=sz)
    plt.rc('axes', labelsize=sz)
    plt.rc('legend', fontsize=14)
    plt.rc('font', size=sz)
    plt.figure(figsize=(width, height))
    plt.plot(to_plot_wl, to_plot, label='Spectrum')
    for k in range(len(args)):
        arg = args[k]
        plt.vlines(arg[0], to_plot.value.min(), to_plot.value.max(), colors=colors[k], label=arg[1])
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel(r'Wavelength (nm)')
    plt.ylabel('Spectrum (ph/s/nm/mode)')
    plt.title(title)
    plt.tight_layout()

def plot_crop_lamp(lamp_wl, wl_bounds, lamp_ph, throughput, title, plot=True):
    mask = (lamp_wl >= wl_bounds[0]) & (lamp_wl <= wl_bounds[1])
    nb_phot = np.trapz(lamp_ph[mask]*throughput, lamp_wl[mask])
    if plot:
        plot_lamp(lamp_wl[mask], lamp_ph[mask]*throughput, title)

    return lamp_wl[mask], lamp_ph[mask]*throughput, nb_phot
    
def black_body(temp, wl, h, c, k_B):
    """
    Power emitted per unit area per unit solid angle per level of energy (channel spectrum)
    in W/m2/sr/m.
    This is also the power through a unit of optical étendue per level of energy.
    """
    return 2 * h * c / wl.to(u.m) * 1/wl.to(u.m)**2 * c/wl.to(u.m)/wl.to(u.m) * 1 / (np.exp(h * c / (wl.to(u.m) * k_B * temp)) - 1)

def sellmeier(wl, u0, u1, u2, u3, u4, u5, A):
    n = u0 * wl**2 / (wl**2 - u3**2) + u1 * wl**2 / (wl**2 - u4**2) + u2 * wl**2 / (wl**2 - u5**2) + A
    n = n**0.5
    return n
    
def star_simulator(wl, Teff, mag):
    pass
    
"""
Spectral boundaries of the instrument
"""
heimdallr = np.array((2.18e-6 - 0.45e-6/2, 2.18e-6 + 0.45e-6/2)) * u.m
heimdallr = np.array((2.18e-6 - 0.2e-6/2, 2.18e-6 + 0.2e-6/2)) * u.m
baldr = np.array((1.6e-6 - 0.2e-6/2, 1.6e-6 + 0.3e-6/2)) * u.m
bifrost = np.array((1.35e-6 - 0.6e-6/2, 1.35e-6 + 0.6e-6/2)) * u.m
nott = np.array((3.75e-6 - 0.5e-6/2, 3.75e-6 + 0.5e-6/2)) * u.m

heimdallr = np.around(heimdallr.to(u.nm))
baldr = np.around(baldr.to(u.nm))
bifrost = np.around(bifrost.to(u.nm))
nott = np.around(nott.to(u.nm))

"""
Throughput of the instruments (VLTI not included)
"""
throughput_hei = 0.5
throughput_bal = 0.5
throughput_bif = 0.08
throughput_nott = 0.94 * 0.61 * 0.1427
# throughput_nott = 0.0000023859
throughput_mmf_to_smf = 0.01
throughput_4beams_creator = 1.
throughput_4beams_creator = 0.022
throughput_vlti = 0.33

"""
Load spectra of Thorlabs lamps.
Thorlabs spectra are in arbitrary unit, the peak is equal to 1.
The wavelength is in nm.
"""
sl202l_power = 1.5e-3 * u.W
sl203l_power = 1.5 * u.W
sl202l = load_lamp('SLS202L_Spectrum.txt', sl202l_power)
# sl203l = load_lamp('SLS203L_Spectrum.txt', sl203l_power)
# sl203l = (sl203l[0][:18828], sl203l[1][:18828])

"""
Throughput of the Thorlabs ZrF4 MMF MZ41L1 https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7840&pn=MZ41L1
"""
zrf4_mmf_length = 1# in metre
zrf4_mmf_attenuation = np.loadtxt('ZrF4_attenuation_db_per_m.txt')
zrf4_mmf_throughput = np.interp(sl202l[0].value, zrf4_mmf_attenuation[:,0]*1000, zrf4_mmf_attenuation[:,1])
zrf4_mmf_throughput *= zrf4_mmf_length
zrf4_mmf_throughput = 10**(-zrf4_mmf_throughput/10)

sl202l_W = sl202l[1] * zrf4_mmf_throughput

"""
Convert the spectrum from W/nm/mode into ph/s/nm/mode
"""
sl202l_ph = convert_Watt_to_photons(sl202l[0], sl202l_W, h, c)
# sl203l_ph = convert_Watt_to_photons(sl203l[0], sl203l[1], h, c)

"""
The specifications of the instrument are given per mode of propagation (area ~ lambda**2)
thus we need to get a spectrum per mode.
The output given by Thorlabs for the SL202L is given all mode included for a total of 1.5 mW.
The fiber reference is given, it is a MMF, we can deduce the number of modes carried
by that fiber wrt the wavelength as number_of_modes ~ V**2/2 for step index or V**2/4 for graded index.
Source: https://www.rp-photonics.com/v_number.html
ZrF4 MMF MZ41L1 https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7840&pn=MZ41L1
"""
wl = sl202l[0].to(u.um).value # im um
d_core = 450 # Core diameter in um
n_core = sellmeier(wl, 0.5522, 0.7483, 1.007, 0.043, 0.113, 16.186, 0.9621)
n_cladding = sellmeier(wl, 0.705674, 0.515736, 2.204519, 0.087503, 0.087505, 23.80739, 1)
NA = np.sqrt(n_core**2 - n_cladding**2)
V = 2 * np.pi * NA * d_core/(wl)
nb_modes = V**2 / 2

sl202l_ph_mode = sl202l_ph / nb_modes

"""
Introduce a blac body model of the thermal background for Nott.
The calibration source has to be brighter than the background to be useful for Nott.
"""
background = black_body(290 * u.K, sl202l[0].to(u.m), h, c, k_B)
background = background.to(u.J / u.s / u.m**2 / u.nm) # W/m2/sr/nm
background = background / (h*c/sl202l[0].to(u.m)) * sl202l[0].to(u.m)**2 # in ph/s/nm/mode

"""
Plot the result
"""
plot_lamp(sl202l[0], sl202l_ph, 'SL202L', (heimdallr.value, 'HEIMDALLR'), (baldr.value, 'Baldr'),
          (bifrost.value, 'BIFROST'), (nott.value, 'NOTT'))

heimdallr_ph = plot_crop_lamp(sl202l[0], heimdallr, sl202l_ph_mode, throughput_hei * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - HEI')
baldr_ph = plot_crop_lamp(sl202l[0], baldr, sl202l_ph_mode, throughput_bal * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - BAL')
bifrost_ph = plot_crop_lamp(sl202l[0], bifrost, sl202l_ph_mode, throughput_bif * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - BIF')
bg_ph = plot_crop_lamp(sl202l[0], nott, background, 0.14*(1-0.59), 'SL202L - NOTT background', False)
nott_ph = plot_crop_lamp(sl202l[0], nott, sl202l_ph_mode, throughput_nott * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - NOTT', False)

print(heimdallr_ph[-1] / (3000 * 50000)) #See Mike's email 2022-10-5, "Selecting internal sources"
print('Number of photons per second across instrument spectral bands:')
print('HEI ', heimdallr_ph[-1])
print('BAL ', baldr_ph[-1])
print('BIF ', bifrost_ph[-1])
print('NOTT', nott_ph[-1])
print('BG  ', bg_ph[-1])
print('Is source brighter than background for Nott?', nott_ph[-1] > bg_ph[-1])
print('NOTT / background ratio', nott_ph[-1].value / bg_ph[-1].value)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
width = 6.528 * 1.5
height = width / 1.618
sz = 16
plt.rc('xtick', labelsize=sz)
plt.rc('ytick', labelsize=sz)
plt.rc('axes', labelsize=sz)
plt.rc('legend', fontsize=14)
plt.rc('font', size=sz)
plt.figure(figsize=(width, height))
plt.plot(nott_ph[0], nott_ph[1], label='NOTT spectrum')
plt.plot(bg_ph[0], bg_ph[1], label='Bg spectrum')
plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel('Spectrum (ph/s/nm/mode)')
plt.title('NOTT and Bg spectra')
plt.tight_layout()
    
plt.close('all')

def get_flux_on_px(wl0, R, bandwidth, px_sampling, flux_wl, flux_ph, wl_bounds):
    global wl_pix, dl_pixel
    unit = flux_wl.unit
    if R > 0:
        dl = wl0 / R
    else:
        dl = wl_bounds[1] - wl_bounds[0]
        
    dl = dl.to(unit)
    nb_channels = np.around(bandwidth / dl)
    nb_pixels = nb_channels * px_sampling
    dl_pixel = bandwidth / nb_pixels
    dl_pixel = dl_pixel.to(unit)
    flux_ph_function = interp1d(flux_wl, flux_ph, kind='quadratic')
    wl_pix = np.arange(wl_bounds[0].value, wl_bounds[-1].value+dl_pixel.value, dl_pixel.value) * unit
    wl_pix = np.around(wl_pix, decimals=9)
    wl_pix = wl_pix[(wl_pix>=wl_bounds[0]) & (wl_pix<=wl_bounds[1])]
    flux_ph_px = np.array([quad(flux_ph_function, wl_pix[i].value, wl_pix[i+1].value)[0] for i in range(wl_pix.size-1)]) # in ph/s/px
    
    return wl_pix[:-1], flux_ph_px, nb_pixels

def get_flux_in_pe(flux, QE, dit):
    flux_pe = flux * QE # e-/s/px
    flux_dit = flux_pe * dit # e-/px
    
    return flux_dit
    
nott_wl0 = 3.75 * u.um
resolving_power = 400
nott_Dl = nott[1] - nott[0]
spectral_sampling = 3 # in px
QE = 0.7
ron = 15 # e-/dit
snr = 5
dit = 1. # in second
spatial_width = 2. # in px
well_depth = 85000 # e-
throughput_4beams_creator = 1.

throughput_nott = 0.94 * 0.19 * 0.1427
nott_ph = plot_crop_lamp(sl202l[0], nott, sl202l_ph_mode, throughput_nott * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - NOTT - HR', False)
bg_ph = plot_crop_lamp(sl202l[0], nott, background, 0.14*(1-0.19), 'SL202L - NOTT background - HR', False)

# nott_source_ph = plot_crop_lamp(sl202l[0], nott, sl202l_ph_mode, 1. * throughput_mmf_to_smf * throughput_4beams_creator, 'SL202L - L band', True)

plt.figure(figsize=(width, height))
plt.plot(nott_ph[0], nott_ph[1], label='NOTT spectrum')
plt.plot(bg_ph[0], bg_ph[1], label='Bg spectrum')
plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel('Spectrum (ph/s/nm/mode)')
plt.title('NOTT and Bg spectra')
plt.tight_layout()


nott_ph_wl, nott_ph_px, nott_wl_nbpx = get_flux_on_px(nott_wl0, resolving_power, nott_Dl, spectral_sampling, nott_ph[0], nott_ph[1], nott)
bg_ph_wl, bg_ph_px, bg_wl_nbpx = get_flux_on_px(nott_wl0, resolving_power, nott_Dl, spectral_sampling, bg_ph[0], bg_ph[1], nott)

nott_ph_sp = np.sum(np.reshape(nott_ph_px, (-1, spectral_sampling)), 1) # in ph/s/spectral channel
bg_ph_sp = np.sum(np.reshape(bg_ph_px, (-1, spectral_sampling)), 1) # in ph/s/spectral channel
nott_wl_sp = np.mean(np.reshape(nott_ph_wl, (-1, spectral_sampling)), 1)
bg_wl_sp = np.mean(np.reshape(bg_ph_wl, (-1, spectral_sampling)), 1)
ron /= QE # in ph
well_depth /= QE # in ph
total_well = well_depth * spatial_width * spectral_sampling

dit_list = np.logspace(-4, -1, 10)
signal_list = []
noise_list = []
saturation_list = []

for dit in dit_list:
    nott_dit = nott_ph_sp * dit # ph/dit/spectral channel
    bg_dit = bg_ph_sp * dit # ph/dit/spectral channel
    bg_photon_noise = bg_dit**0.5
    total_noise = (bg_photon_noise**2 + nott_dit*0 + spatial_width * spectral_sampling * ron**2)**0.5
    total_flux = nott_dit + bg_dit
    saturation = np.zeros_like(nott_dit, dtype=bool)
    saturation[total_flux > 0.9*total_well] = True
    
    signal_list.append(nott_dit)
    noise_list.append(total_noise)
    saturation_list.append(saturation)

signal_list = np.array(signal_list)
noise_list = np.array(noise_list)
saturation_list = np.array(saturation_list)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
width = 6.528 * 1.5
height = width / 1.618
sz = 16
plt.rc('xtick', labelsize=sz)
plt.rc('ytick', labelsize=sz)
plt.rc('axes', labelsize=sz)
plt.rc('legend', fontsize=14)
plt.rc('font', size=sz)
plt.figure(figsize=(width, height))
plt.semilogy(nott_wl_sp, signal_list.T, '-', lw=3)
plt.semilogy(bg_wl_sp, noise_list.T, '--', lw=3)
plt.grid()
plt.legend(['DIT = %.1g s'%elt for elt in dit_list], loc='upper left', bbox_to_anchor=[1. ,1.])
plt.xlabel(r'Wavelength (nm)')
plt.ylabel('Spectrum (ph/DIT/spectral channel)')
plt.title('NOTT and Bg spectra')
plt.tight_layout()
# plt.savefig('/mnt/96980F95980F72D3/Asgard/acau/flux_acau_throughput_%.1f_R_%04d.png'%(throughput_4beams_creator*100, resolving_power), format='png', dpi=150)

plt.figure(figsize=(width, height))
plt.semilogy(nott_wl_sp, signal_list.T/noise_list.T, '-', lw=3)
plt.semilogy(nott_wl_sp, [snr]*nott_wl_sp.size, 'k--', lw=3)
plt.grid()
plt.legend(['DIT = %.1g s'%elt for elt in dit_list]+['Required SNR'], loc='upper left', bbox_to_anchor=[1. ,1.])
plt.xlabel(r'Wavelength (nm)')
plt.ylabel('SNR')
plt.title('NOTT SNR per spectral channel')
plt.tight_layout()
# plt.savefig('/mnt/96980F95980F72D3/Asgard/acau/snr_acau_throughput_%.1f_R_%04d.png'%(throughput_4beams_creator*100, resolving_power), format='png', dpi=150)

plt.figure(figsize=(width, height))
plt.plot(nott_wl_sp, saturation_list.T, '.-', lw=3)
plt.grid()
plt.legend(['DIT = %.1g s'%elt for elt in dit_list]+['Required SNR'], loc='upper left', bbox_to_anchor=[1. ,1.])
plt.xlabel(r'Wavelength (nm)')
plt.ylabel('Saturation')
plt.title('Saturation')
plt.tight_layout()
plt.tight_layout()

total_noise_ph = (bg_ph[1].value + nott_ph[1].value + spatial_width * spectral_sampling * ron**2)**0.5

plt.figure(figsize=(width, height))
plt.plot(nott_ph[0].value, total_noise_ph*5, '-', lw=3)
plt.grid()
plt.xlabel(r'Wavelength (nm)')
plt.tight_layout()
plt.tight_layout()