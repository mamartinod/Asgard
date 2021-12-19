"""
If the input pupil is 
"""
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#In mm
beam_size = 10.0 
#Longest wavelength (in microns)
wave = 3.0
#Gaussian 1/e^2 half-width in units of lambda/D
gw = 0.70
obstruction_sz = 0.1
#------------------------------
r = np.arange(300)/100
difflim_E = ot.airy(r, obstruction_sz=obstruction_sz)
#Gaussian (rough waveguide mode)
gg = np.exp(-(r/gw)**2)
#2D radially symmetric integral normalisation
norm = np.trapz(gg**2*r,r)*np.trapz(difflim_E**2*r, r)
zs = np.arange(21)/2 * 1000
overlaps = np.empty(21)
for i, z in enumerate(zs):
    complex_integrand = r*gg*difflim_E*np.exp(-1j*np.pi*r**2*wave*1e-3*z/beam_size**2)
    overlaps[i] =  (np.trapz(complex_integrand.real,r)**2 + np.trapz(complex_integrand.imag,r)**2)/norm 

plt.figure(1)
plt.clf()
plt.plot(zs/1000, overlaps)
plt.ylabel('Overlap Integral')
plt.xlabel('Fresnel Diffraction Distance')
plt.tight_layout()
plt.savefig('../images/fresnel_loss.png')