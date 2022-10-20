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
throughput_nott = 0.0837
# throughput_nott = 0.0000023859
throughput_mmf_to_smf = 0.01

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

heimdallr_ph = plot_crop_lamp(sl202l[0], heimdallr, sl202l_ph_mode, throughput_hei * throughput_mmf_to_smf, 'SL202L - HEI')
baldr_ph = plot_crop_lamp(sl202l[0], baldr, sl202l_ph_mode, throughput_bal * throughput_mmf_to_smf, 'SL202L - BAL')
bifrost_ph = plot_crop_lamp(sl202l[0], bifrost, sl202l_ph_mode, throughput_bif * throughput_mmf_to_smf, 'SL202L - BIF')
nott_ph = plot_crop_lamp(sl202l[0], nott, sl202l_ph_mode, throughput_nott * throughput_mmf_to_smf, 'SL202L - NOTT')


bg_ph = plot_crop_lamp(sl202l[0], nott, background, 0.14*(1-0.59), 'SL202L - NOTT background')

print(heimdallr_ph[-1] / (3000 * 50000)) #See Mike's email 2022-10-5, "Selecting internal sources"
print('Number of photons per second across instrument spectral bands:')
print('HEI ', heimdallr_ph[-1])
print('BAL ', baldr_ph[-1])
print('BIF ', bifrost_ph[-1])
print('NOTT', nott_ph[-1])
print('BG  ', bg_ph[-1])
print('Is source brighter than background for Nott?', nott_ph[-1] > bg_ph[-1])
print('NOTT / background ratio', nott_ph[-1].value / bg_ph[-1].value)
