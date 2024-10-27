"""
Heimdallr will be a Fringe tracker and alignment tool for all of Asgard.

This script optimises the focal ratio of Heimdallr for signal to noise on the longest
baselines in a case limited by pixel-based noise. This noise comes from either readout 
noise, or thermal background from the imperfect Narcissis mirrors. The baseline design 
(Heimdallr_K.pdf) has 11 surfaces at (optimistically)~1% per surface, means that 11%
of the background is seen in spite of the Narcissis mirror. This is also the rough 
fractional area of the 4 input apertures. 
"""

import numpy as np

pix_sz = 24e-6
#Longest possible wavelength. Should notionally stop (50% filter cutoff) at 2.35 microns
wave = 1.9e-6 
short_bl_on_diam = 2.2
long_bl_on_diam = np.sqrt(3) * short_bl_on_diam
max_frac = long_bl_on_diam/(long_bl_on_diam+1)
print("Maximum allowable Nyquist fracton: {:.2f}".format(max_frac))

#Nyquist fraction of longest baseline
nyquist_frac = np.linspace(0.5,max_frac,100)

#Background-limited SNR is proportional to the aperture area as a fraction of the
#cold stop area.
bg_snr = np.sinc(0.5*nyquist_frac)*nyquist_frac**2

best_nyquist_frac = nyquist_frac[np.argmax(bg_snr)]
print("Optimal Nyquist fraction for Long Baseline: {:.2f}".format(best_nyquist_frac))
print("Nyquist frac for short baselines: {:.3f}".format(best_nyquist_frac*short_bl_on_diam/long_bl_on_diam))
print("Nyquist frac for single aperture: {:.3f}".format(best_nyquist_frac/long_bl_on_diam))

nyquist_angle = np.degrees(wave/(2*pix_sz))
nyquist_frat = (2*pix_sz)/wave
print("Angle for Nyquist: {:.3f} degrees".format(nyquist_angle))
print("Convergence Angle: {:.3f} degrees".format(nyquist_angle * best_nyquist_frac*short_bl_on_diam/long_bl_on_diam))
single_beam_frat = nyquist_frat*long_bl_on_diam/best_nyquist_frac
print("Single Beam focal-ratio: {:.1f}".format(single_beam_frat))

#Now check what this means for the cold stop
perfect_cstop_frat = single_beam_frat/(short_bl_on_diam*2 + 1)
real_cstop_frat = single_beam_frat/(short_bl_on_diam*2 + 2)
print("Ideal Maximum Cold Stop F-ratio: {:.1f}".format(perfect_cstop_frat))
print("Realistic Maximum Cold Stop F-ratio: {:.1f}".format(real_cstop_frat))