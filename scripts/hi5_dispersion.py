"""Determine the ability for VLTI to have simultaneous near-infrared fringes in 
several bandpasses

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


#Air properties. 
plot_extra=False
t_air = 5.0 #InC
p_air = 750e2 #In Pascals

#From https://www.eso.org/gen-fac/pubs/astclim/paranal/h2o/
#there is between 0.05 and 0.1 mol/m^3
#From https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-air-d_689.html
#... there is ~800 Pa of partial pressure at 100% humidity at 5C, which is 0.34 mol/m^3.
#... meaning that a humidity of 30% corresponds to that 0.1 mol/m^3 line. 

h_air = 0.3 #humidity: between 0 and 1
glass = 'al2o3'
glass2 = 'znse' 
delta = 50.0
N_wn = 100

#Wave-number in um^-1
wn = np.linspace(1/4.0,1/3.4,N_wn)
offset = -25e-6/100
#wn = np.linspace(1/3.9,1/3.5,N_wn)
#offset = -12e-6/100

mn_wn = 0.5*(wn[1:] + wn[:-1])
del_wn = wn[1:] - wn[:-1]

nm1_air = ot.nm1_L_air(1./wn,t_air,p_air,h_air)
n_glass = ot.nglass(1./wn, glass=glass)
n_glass2 = ot.nglass(1./wn, glass=glass2)

#Derivatives evaluated everywhere but endpoints.
d1_air = (nm1_air[2:]-nm1_air[:-2])/(wn[2:] - wn[:-2]) 
d2_air = (nm1_air[2:]-2*nm1_air[1:-1]+nm1_air[:-2])/(0.5*(wn[2:] - wn[:-2]))**2
b1_air = 1.0 + nm1_air[1:-1] + wn[1:-1] * d1_air 
b2_air = d1_air + 1/2. * wn[1:-1] * d2_air

d1_glass = (n_glass[2:]-n_glass[:-2])/(wn[2:] - wn[:-2]) 
d2_glass = (n_glass[2:]-2*n_glass[1:-1]+n_glass[:-2])/(0.5*(wn[2:] - wn[:-2]))**2
b1_glass = n_glass[1:-1] + wn[1:-1] * d1_glass
b2_glass = d1_glass + 1/2. * wn[1:-1] * d2_glass

b_arrs = np.array([[b1_air,b1_glass],[b2_air,b2_glass]])
b_arrs = b_arrs.transpose( (2,0,1) )
x0s = np.zeros( (len(b1_air),2) )
x0s[:,0] = 1.0
x_matsolve = np.linalg.solve(b_arrs, x0s)
#---
wn_cent = 0.5*(wn[1:-2]+wn[2:-1])
#The same, but for 2 glasses. Evaluate derivatives at mid-points.
d1_air = (nm1_air[1:]-nm1_air[:-1])/(wn[1:] - wn[:-1])      #1 shorter
d2_air = (d1_air[1:]-d1_air[:-1])/(0.5*(wn[2:] - wn[:-2]))  #2 shorter
d3_air = (d2_air[1:]-d2_air[:-1])/(wn[2:-1] - wn[1:-2])     #3 shorter
#Make all centered the same way...
d1_air = d1_air[1:-1]
d2_air = 0.5*(d2_air[1:] + d2_air[:-1])
b1_air = 1.0 + 0.5*(nm1_air[1:-2]+nm1_air[2:-1]) + wn_cent * d1_air 
b2_air = d1_air + 1/2. * wn_cent * d2_air
b3_air = 1/2.*d2_air + 1/6. * wn_cent * d3_air

d1_glass = (n_glass[1:]-n_glass[:-1])/(wn[1:] - wn[:-1])      #1 shorter
d2_glass = (d1_glass[1:]-d1_glass[:-1])/(0.5*(wn[2:] - wn[:-2]))  #2 shorter
d3_glass = (d2_glass[1:]-d2_glass[:-1])/(wn[2:-1] - wn[1:-2])     #3 shorter
#Make all centered the same way...
d1_glass = d1_glass[1:-1]
d2_glass = 0.5*(d2_glass[1:] + d2_glass[:-1])
b1_glass = 0.5*(n_glass[1:-2]+n_glass[2:-1]) + wn_cent * d1_glass 
b2_glass = d1_glass + 1/2. * wn_cent * d2_glass
b3_glass = 1/2.*d2_glass + 1/6. * wn_cent * d3_glass

d1_glass2 = (n_glass2[1:]-n_glass2[:-1])/(wn[1:] - wn[:-1])      #1 shorter
d2_glass2 = (d1_glass2[1:]-d1_glass2[:-1])/(0.5*(wn[2:] - wn[:-2]))  #2 shorter
d3_glass2 = (d2_glass2[1:]-d2_glass2[:-1])/(wn[2:-1] - wn[1:-2])     #3 shorter
#Make all centered the same way...
d1_glass2 = d1_glass2[1:-1]
d2_glass2 = 0.5*(d2_glass2[1:] + d2_glass2[:-1])
b1_glass2 = 0.5*(n_glass2[1:-2]+n_glass2[2:-1]) + wn_cent * d1_glass2 
b2_glass2 = d1_glass2 + 1/2. * wn_cent * d2_glass2
b3_glass2 = 1/2.*d2_glass2 + 1/6. * wn_cent * d3_glass2

b_arrs = np.array([[b1_air,b1_glass,b1_glass2],[b2_air,b2_glass,b2_glass2],[b3_air,b3_glass,b3_glass2]])
b_arrs = b_arrs.transpose( (2,0,1) )
x0s = np.zeros( (len(b1_air),3) )
x0s[:,0] = 1.0
x2_matsolve = np.linalg.solve(b_arrs, x0s)
#---


#The best solution at the midpoint is a reasonable solution, but not quite perfect. 
#No point doing the least-squares as we have to tweak anyway for the null. Lets make a plot.
x0 = x_matsolve[4*N_wn//10] 
x0_2 = x2_matsolve[4*N_wn//10] 
phase = 2*np.pi*delta*1e6*((x0[0] + offset)*(nm1_air+1.0) + x0[1]*n_glass - 1.0)*wn
phase2 = 2*np.pi*delta*1e6*(x0_2[0]*(nm1_air+1.0) + x0_2[1]*n_glass +x0_2[2]*n_glass2 - 1.0)*wn
plt.figure(1)
#plt.clf()
plt.plot(1/wn, phase-np.mean(phase), 'C3', label='1 glass')
plt.plot(1/wn, phase2-np.mean(phase2), 'C2', label='2 glasses')
plt.legend()

plt.xlabel('Wavelength')
plt.ylabel(r'Fringe Phase (radians)')
plt.title('{0:5.1f}m of air path and 30% RH'.format(delta))

print('Glass thickness: {:5.2f}mm'.format(x0[1]*delta*1e3))
print('Double glass thicknesses: {:5.2f}mm, {:5.2f}mm'.format(x0_2[1]*delta*1e3, x0_2[2]*delta*1e3))

