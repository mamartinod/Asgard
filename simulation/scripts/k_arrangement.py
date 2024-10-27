"""Arrange the k-band beams.

Also check the expected background on the detector.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pdb
plt.ion()

#-----   1st section ---  
def planck_lambda(wave, T=290):
    """
    Input Wavelength in microns, and flux per unit micron. Area also in 
    microns^2
    
    Multiply by solid angle and area to get flux.
    
    Sanity check:
    Convert per unit frequency to wavelength by a c/lambda^2 factor
    Convert from per mode to per unit area by dividing by lambda^2
    """
    return 1e-6**3*2*3e8/(wave*1e-6)**4/(np.exp(6.626e-34*3e8/(wave*1e-6)/1.38e-23/T)-1)

wave1 = np.linspace(1.95,2.15,20)
wave2 = np.linspace(2.15,2.35,20)
f_ratio = 20
qe = 0.8
emiss1 = 0.34 #Emissivity in the target wavelength
emiss2 = 0.14 #Emissivity in the other wavelength band

elec_sec1 = np.trapz(planck_lambda(wave1),wave1)*24**2*np.pi*(0.5/f_ratio)**2*qe
elec_sec2 = np.trapz(planck_lambda(wave2),wave2)*24**2*np.pi*(0.5/f_ratio)**2*qe
k1_background = emiss1 * elec_sec1 + emiss2 * elec_sec2
k2_background = emiss2 * elec_sec1 + emiss1 * elec_sec2
print("K1 background (elec/s): {:.1f}".format(k1_background))
print("K2 background (elec/s): {:.1f}".format(k2_background))

#-----   2nd section ---  
    
def rotate_mat(theta):
    #Clockwise rotation of vectors.
    mat = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    return mat

#Mount separation in mm, for 26mm mounts.
mount_sep = 30.

#Separation between the bandpasses on the detector
bandpass_sep = 0.024*160

#Pivot to detector 
pivot_detector_dist = 140.

#Beam_angles, left to right.
beam_angles = [bandpass_sep/2/38.83,-bandpass_sep/2/38.83]
#np.radians([-6,6])

#The beam reflected off the first dichroic
refl_vect = rotate_mat(np.radians(50)).dot([0,1])

#The beam reflected off the mirror
refl_vect2 = rotate_mat(np.radians(61.5)).dot([0,1])

#A hack scaling factor for the prism
alpha=1.3

#Angle of the pivot
npivot = 1000
pivot_angle = np.linspace(0,3,1000)

#Lets call the pivot point [0,0], and trace the position of b0.
#Once we have:
#d1 * u1 + x1 = d2 * u2 + x2
#... for scalars d1 an d2, and vectors x1 and x2, this becomes:
# [u1 -u2] . [d1,d2]^T = x2-x1, solvable as a matrix equation.
pivot_det_b3_vect = np.array([rotate_mat(th).dot(refl_vect) for th in pivot_angle])
det_vect = np.array([rotate_mat(np.pi/2-beam_angles[1]).dot(p) for p in pivot_det_b3_vect])
det_b0 = pivot_detector_dist * pivot_det_b3_vect + bandpass_sep*det_vect 
m1 = -alpha*mount_sep*refl_vect             #First Dichroic.
m4 = m1 -alpha*mount_sep*refl_vect*refl_vect[1] - 10*refl_vect
m2 = m1 - mount_sep*np.array([0,1])     #Fold mirror
det_b0_m3_vect = np.array([rotate_mat(beam_angles[0]-beam_angles[1]).dot(-p) for p in pivot_det_b3_vect])
m3 = []
b0_dists = []
u1 = refl_vect2
x1 = m2
for u2,x2 in zip(det_b0_m3_vect, det_b0):
    d1d2 = np.linalg.solve(np.array([u1,-u2]).T, x2-x1)
    m3.append(d1d2[0]*u1 + x1)
    b0_dists.append(np.sum(d1d2))
b0_dists=np.array(b0_dists)
m3 = np.array(m3)
b0_dists += mount_sep
b3_dist = pivot_detector_dist + alpha*mount_sep

#Now find the best angle and hence best position.
ix = np.argmin(np.abs(b3_dist - b0_dists))
print("Pivot angle (deg): {:5.1f}".format(np.degrees(pivot_angle[ix])))
print("Minimum distance (mm): {:5.2f}".format(np.abs(b3_dist - b0_dists)[ix]))
b0 = np.array([m1,[0,0],pivot_det_b3_vect[ix]*pivot_detector_dist]).T
b3 = np.array([[m1[0],20],m1,m2,m3[ix],det_b0[ix]]).T
plt.clf()

#Finally, some mirror surfaces...
mirrors = [m1,m2] #,m4,m5,m7]
labels = ['D1', 'M'] #, 'D2','D3','D4']
prisms = [m3[ix],[0,0]] #[m3[ix],m6,m8,[0,0]]
mirror_surface_vect = 12.7*rotate_mat(np.radians(25)).dot([1,0])
prism_surface_vect =  5*rotate_mat(np.radians(25)).dot([1,0])
prism_corner_vect =  5*rotate_mat(np.radians(25) - np.pi/2).dot([1,0])

for m, l in zip(mirrors, labels):
    plt.plot([m[0]-mirror_surface_vect[0],m[0]+mirror_surface_vect[0]],\
        [m[1]-mirror_surface_vect[1],m[1]+mirror_surface_vect[1]], 'k')
    plt.text(m[0]+6,m[1],l)

for p in prisms:
    plt.plot([p[0]-prism_surface_vect[0],p[0]+prism_surface_vect[0], \
              p[0]+prism_corner_vect[0], p[0]-prism_surface_vect[0]],\
        [p[1]-prism_surface_vect[1],p[1]+prism_surface_vect[1], \
         p[1]+prism_corner_vect[1], p[1]-prism_surface_vect[1]], 'k')

#plt.plot([3*det_b3[0]-2*det_b0[0], 3*det_b0[0]-2*det_b3[0]], [3*det_b3[1]-2*det_b0[1], 3*det_b0[1]-2*det_b3[1]],'k')
plt.plot([2*b3[0,-1]-1*b0[0,-1], 2*b0[0,-1]-1*b3[0,-1]], [2*b3[1,-1]-1*b0[1,-1], 2*b0[1,-1]-1*b3[1,-1]],'k')

#Plot the cold shield
det_cent = 0.5*(b3[:,-1] + b0[:,-1])
shield_vect = rotate_mat(np.pi/2).dot(det_vect[ix])
shield = np.array([[-1,-8,-8,8,8,1],[39,39,-1,-1,39,39]])
plt.plot(shield[0]+det_cent[0], shield[1]+det_cent[1], 'b')
enclosure = np.array([[-63,-63,-20,-20,20,20,100],[-100,48,48,57,57,48,48]])
plt.plot(enclosure[0]+det_cent[0], enclosure[1]+det_cent[1], 'b:')

#The cold stops
plt.plot([-15+m1[0],15+m1[0]],[10,10], 'k')
plt.text(8+m1[0],5,'C')
stop_surface_vect = 15*rotate_mat(np.radians(50)).dot([1,0])
plt.plot([-stop_surface_vect[0]+m4[0],stop_surface_vect[0]+m4[0]],\
[-stop_surface_vect[1]+m4[1],stop_surface_vect[1]+m4[1]], 'k')
plt.text(m4[0]-6,m4[1]+10,'C')

#The mount for the H dichroic.
theta = np.linspace(0,2*np.pi,100)
Hdich_center = det_cent + shield_vect*68
mount_rad=16.5/2
plt.plot(Hdich_center[0] + mount_rad*np.cos(theta), Hdich_center[1] + mount_rad*np.sin(theta),'m:')


#The beams
#plt.plot(b2[0], b2[1], label='H1')
#plt.plot(b1[0], b1[1], label='H2')
plt.plot(b0[0], b0[1], 'r', label='K1')
plt.plot(b3[0], b3[1], 'g', label='K2')
plt.plot([b0[0,0], m4[0]], [b0[1,0], m4[1]], 'g:')


plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.legend()
#plt.axes().set_aspect('equal')
plt.axis([-75,30,-150,15])

#Make a scale bar
#plt.plot(np.array([-35,-35,-35,-15,-15,-15])-5,np.array([-122,-116,-119,-119,-116,-122])+40,'k')
#plt.text(-12-5,-120+40,'20mm')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

#plt.axis('off')

plt.tight_layout()