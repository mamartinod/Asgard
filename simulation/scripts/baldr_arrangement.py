from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import root_scalar
plt.ion()

def rotate_mat(theta):
    #Clockwise rotation of vectors.
    mat = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    return mat
    
def trig_func(theta, ratio):
    return np.sin(theta)/(1+np.cos(theta)) - ratio


#Minimum beam separation.
min_beam_sep = 43.9

#Initial (VLTI) beam separation
init_beam_sep = 240.

#Extra y separation
extra_y = 600

#Cold Stop Distance
cold_dist = 39

#Camera Focal Length
f_cam = 200 #LA1708-C

#Pupil size on camera
cam_pup_size = 0.024*11.5 #7.67

#Is the second mirror an OAP?
second_mirror_oap = True

#Input y coord (top of table)
input_yoff = 400

#Output x coord (left of table)
output_xcoord = -200

#Initial (collimated) pupil size
init_pup_size = 12

#For a ~1mm pupil and F/10, this works for a compact lens.
f_collimator = 30 #LA1289-C (Was LA1074-C)

#-------
#On-axis calcs. 
pup_dist = 1/(1/f_cam - 1/(f_cam + cold_dist))
print("Distance from pupil to lens (check > 1000mm): {:.1f}".format(pup_dist))
pup_mag = pup_dist/(f_cam + cold_dist)
intermediate_pup_size = cam_pup_size * pup_mag
zwfs_demag = intermediate_pup_size/init_pup_size
f_oap = f_collimator/zwfs_demag
print("OAP Focal length: {:.1f}".format(f_oap))

#Beam positions
input_xpos = init_beam_sep*np.arange(4)[::-1]
c4 = 2*init_beam_sep + 2*min_beam_sep
ratio = (init_beam_sep - 2*min_beam_sep)/c4
th = root_scalar(trig_func, args=(ratio,), bracket=[0,1]).root

print("Primary angle: {:.2f}".format(np.degrees(th)))
print("Half angle: {:.2f}".format(np.degrees(th)/2))

xseps = np.linspace(min_beam_sep, init_beam_sep-min_beam_sep, 4)
output_xpos = xseps + input_xpos
bs = xseps/np.tan(th)
a_s = xseps/np.sin(th)

if second_mirror_oap:
    dogleg_yoff = output_xpos + bs - output_xpos[-1]-bs[-1]
else:
    dogleg_yoff = np.zeros(4)

print("Right to left...")
print("L1s: " + str(dogleg_yoff + input_yoff))
print("L2s: " + str(a_s + 127**2/4/254))
l3s = bs-dogleg_yoff + extra_y 
print("L3s: " + str(l3s))
l3s_mod = l3s - 254-17.3-3.98
print("L3s (mod): " + str(l3s_mod))
print("L4s: " + str(output_xpos - output_xcoord))
l4s_mod = pup_dist - l3s_mod
print("L4s (mod): " + str(l4s_mod))
print("OAP: " + str(127**2/4/254/np.cos(np.radians(30)) +  254 + 17.3+3.98))

mat = rotate_mat(th/2)
mat45 = rotate_mat(-np.radians(45))
mirror_xy = np.array([[-12.5,12.5],[0,0]])
plt.clf()
plt.axes().set_aspect('equal')
for i in range(4):
    plt.plot( [input_xpos[i], input_xpos[i], output_xpos[i], output_xpos[i], output_xcoord], \
        [input_yoff, -dogleg_yoff[i], bs[i]-dogleg_yoff[i], -extra_y, -extra_y], 'C{:d}'.format(i))
    xy = np.dot(mat, mirror_xy)
    plt.plot(xy[0] + input_xpos[i], xy[1]-dogleg_yoff[i], 'k')
    plt.plot(xy[0] + output_xpos[i], xy[1] + bs[i]-dogleg_yoff[i], 'k')
    xy = np.dot(mat45, mirror_xy)
    plt.plot(xy[0] + output_xpos[i], xy[1] - extra_y, 'k')
plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.tight_layout()