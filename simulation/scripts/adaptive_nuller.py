"""
Create a point design for an adaptive nuller. Also consider cross-talk.

As an input, Azzura uses a 646mm effective focal length Edmund optics mirror with 
lambda/2 RMS reflected WFE. This is definitely a potential issue, even though the 
sub-aperture is only 12mm. In turn, this has an effective focal ratio of 53.8. 

These beams have a separation of 1.333mm. My requested separation is 0.6 times this, or 
0.8mm. There is also a concern that the focal ratio of the beams isn't quite optimal. 
At 3.7 microns, I definitely got an optimal that meant either a focal ratio smaller 
by 0.91, or a separation larger by 1/0.91.


This has a radius of 167mm on the lathe, which is a but much for ANU.  A focal length of 
353mm is roughly maximum. 

Assumption: We want to match the Thorlabs better spec OAP with:
101.6mm PFL
203.2mm RFL

Important Notes
---------------
1) The DM should be (slightly) tilted vertically, to increase the range for adjustment
of flux.
2) The ~77% maximum coupling is an issue best solved by PIAA. It can also be solved
by an input mask.
3) Cold stop may not be in input pupil plane?

203.2 + 
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import pdb
import sys
plt.ion()
np.set_printoptions(precision=5)
if not '..' in sys.path:
    sys.path.insert(0,'..')
import opticstools_abridged as ot
import scipy.optimize as op

#First, a simple calculation about the cross-talk. It is only relevant to a sharp-edged
#pupil with a sharp secondary obstruction. Otherwise a factor of ~10 lower.
pitch_wg=125
microns_pix = 2
sz = 512
#A pupil that gives a 20 micron PSF
lambda_d_microns = 17.0
psize = 512/(lambda_d_microns/microns_pix)
pup = ot.circle(sz, psize, interp_edge=True) - ot.circle(sz, psize/8, interp_edge=True)
pup = ot.circle(sz, psize, interp_edge=True) 
psf_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup)))
fibmode =ot.mode_2d(2, 9, sampling=2,  sz=512)

coupling = np.abs(np.sum(fibmode*psf_E.conj()))**2/np.sum(np.abs(fibmode)**2)/np.sum(np.abs(psf_E)**2)
print("On-axis coupling: {:.3f}".format(coupling))
mask = ot.circle(sz, 2.5*lambda_d_microns/microns_pix, interp_edge=True)
coupling_masked = np.abs(np.sum(mask*fibmode*psf_E.conj()))**2/np.sum(np.abs(fibmode)**2)/np.sum(np.abs(psf_E)**2)
print("On-axis masked coupling: {:.3f}".format(coupling_masked))
moff1 = np.roll(fibmode,int((pitch_wg+1)/microns_pix))
moff2 = np.roll(fibmode,int((pitch_wg-1)/microns_pix))
coupling1 = np.abs(np.sum(moff1*psf_E.conj()))**2/np.sum(np.abs(fibmode)**2)/np.sum(np.abs(psf_E)**2)
coupling2 = np.abs(np.sum(moff2*psf_E.conj()))**2/np.sum(np.abs(fibmode)**2)/np.sum(np.abs(psf_E)**2)
print("Off-axis coupling: {:.2e} {:.2e}".format(coupling1,coupling2))

#Next, lets define the focal ratio for the 800 micron pitch.
pitch_dm=800
wave_microns = 3.7
f_ratio = pitch_dm/pitch_wg * lambda_d_microns/wave_microns
print("Nominal Focal Ratio: {:.1f}".format(f_ratio))

#Based on this, a 3.5 micron stroke and a 20% interactuator coupling, lets detrmine
#the maximum flux loss
magnification = pitch_dm/pitch_wg
radians_per_micron = 3.5*2*.8/3.7*2*np.pi/400
x_tilt = (np.arange(sz)-sz//2) * microns_pix * radians_per_micron * magnification
xy_tilt = np.meshgrid(x_tilt,x_tilt)
tilted_coupling = np.abs(np.sum(fibmode*np.exp(1j*xy_tilt[0])*psf_E.conj()))**2/\
    np.sum(np.abs(fibmode)**2)/np.sum(np.abs(psf_E)**2)
print("On-axis tilted coupling: {:.3f}".format(tilted_coupling))

#Figure out which glass to use for dispersion
glasses = ['mgf2','caf2','al2o3','znse']
ndiffs=[]
for g in glasses:
    ns = ot.nglass([3.4,4], g)
    ndiffs += [ns[0]-ns[1]]
glass = glasses[np.argmax(ndiffs)]
print(glass)

#Figure out the required pupil diameter for a 40 degree prism.
spectrum_length = 3.6
#prism_angle = np.radians(37.3) #for 200mm
prism_angle = np.radians(36.85)
wave = np.linspace(3.4,4,10)
ns = ot.nglass(wave,glass)
input_angle = np.arcsin(ns*np.sin(prism_angle/2))
print("Prism glass input angle in degrees: {:.2f}".format(np.degrees(input_angle[5])))
deviation = 2*input_angle - prism_angle

flength = spectrum_length/(deviation[0]-deviation[-1])
print("Focal length: {:.2f}".format(flength))
print("Pupil diameter: {:.2f}".format(flength/f_ratio))

#The effective focal length of an off-axis section is:
#2 f_p/(1 + cos(theta)) = R_p / (1 + cos(theta))
#We'd probably have a custom mirror of 100mm diameter, with footprints centered
#+/-35mm fron the center.
theta = np.arcsin(35/200)
f_p = (1+np.cos(theta))/2*200
print("Parent focal length: {:.2f}".format(f_p))
print("z offset (add to 203.6): {:.2f}".format(35**2/f_p))
