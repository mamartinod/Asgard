import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
plt.ion()

dx = 0.05 	#dx in units of lambda/D
sz = 256	#Array size for 2D integration
fwhm = 1.2 	#Electric field FWHM in units of Airy radius

#-------------------------------
nshifts = int(2/dx) #The number of shifts we want to calculate (up to 2 lambda/D)
sigma_gauss = fwhm/2.355 #Gaussian standard deviation.

#Make a 2-dimensional grid of radius and xy.
x = (np.arange(sz) - sz//2) * dx
xy = np.meshgrid(x,x)
rr = np.sqrt(xy[0]**2 + xy[1]**2)

#For convenience, make an airy disk function
def airy(x):
	x_trunc = np.maximum(x, 1e-10)
	return j1(x_trunc*np.pi)/(x_trunc*np.pi)
	
#Create our Gaussian electric field
gg = np.exp(-(rr**2/2/sigma_gauss**2))

#Create our Airy disk
aa = airy(rr)

#Now make the overlap integrals!
overlaps = np.zeros(nshifts)
norm = np.sum(np.abs(aa**2))*np.sum(np.abs(gg)**2)
for i in range(0,nshifts):
	overlaps[i] = np.abs(np.sum(aa*np.roll(gg, (i,0))))**2 / norm

#!!! NB, a this point, there are lots of images (gg and aa) to plot
#and sanity-check...

#Finally, make the coupling plot.
shifts = dx * np.arange(nshifts)
plt.clf()
plt.plot(shifts, overlaps)
plt.xlabel(r'Shift ($\lambda$/D)')
plt.ylabel('Coupling')
print("Maximum coupling: {:.2f}".format(np.max(overlaps)))
	