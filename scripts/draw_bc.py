"""
Draw a simple beam compressor in python.

The "back" of the AMBER table is the coordinate system (0,0,0) point. 

This could be a starting point for input into Zemax, similar to what Barnaby did
for the Heimdallr initial proposal.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#Dimensions in mm.
kTableWidth = 1500.
kTableThick = 100. #Just for drawing purposes - it is actually thicker.
kVLTIBeamWidth = 18.
kAMBERHeight = 200 #Beam height on the amber table
kPupilLocation = kTableWidth + 2510.

kBeamWidth = 12.
kDMSize = 4.4
kOAP1x = 100 #Hard to get closer to the table edge than this.
kAsgardHeight = 125 #Beam height chosen for Asgard.

#----
#First, calculate the location of the DM
DM_magnification = kDMSize/kVLTIBeamWidth
OAP1_DM_r = DM_magnification * (kPupilLocation - kOAP1x)
OAP1_DM_theta = np.arcsin((kAsgardHeight - kAMBERHeight)/OAP1_DM_r)
DMx = OAP1_DM_r*np.cos(OAP1_DM_theta) + kOAP1x
DMy = OAP1_DM_r*np.sin(OAP1_DM_theta) + kAMBERHeight

#Now the focal length of OAP1
OAP1_f = 1/(1/(kPupilLocation - kOAP1x) + 1/OAP1_DM_r)
#The beam expands from the focus, reflecting off the planar DM to OAP2
focus_to_DM = OAP1_DM_r - OAP1_f
OAP2_f = focus_to_DM*(kBeamWidth/kDMSize)
DM_OAP2_r =  OAP2_f - focus_to_DM
OAP2_x = DMx - DM_OAP2_r
OAP2_y = kAsgardHeight


plt.figure(1)
plt.clf()
plt.plot([0,kTableWidth, kTableWidth, 0,0], [0,0,-kTableThick, -kTableThick,0], 'k', linewidth=2)

#plt.plot([kTableWidth, kOAP1x, DMx, OAP2_x], [kAMBERHeight, kAMBERHeight, DMy, OAP2_y], 'r')
plt.plot([kTableWidth, kOAP1x, DMx, OAP2_x], [kAMBERHeight+kVLTIBeamWidth/2, kAMBERHeight+kVLTIBeamWidth/2, DMy-kDMSize/2, OAP2_y-kBeamWidth/2], 'r')
plt.plot([kTableWidth, kOAP1x, DMx, OAP2_x], [kAMBERHeight-kVLTIBeamWidth/2, kAMBERHeight-kVLTIBeamWidth/2, DMy+kDMSize/2, OAP2_y+kBeamWidth/2], 'r')

#Now plot the mirrors
plt.plot([kOAP1x-np.sin(OAP1_DM_theta/2)*kVLTIBeamWidth/2, kOAP1x+np.sin(OAP1_DM_theta/2)*kVLTIBeamWidth/2], \
    [kAMBERHeight+kVLTIBeamWidth/2, kAMBERHeight-kVLTIBeamWidth/2], 'k', linewidth=2)
plt.plot([DMx-np.sin(OAP1_DM_theta/2)*kDMSize/2, DMx+np.sin(OAP1_DM_theta/2)*kDMSize/2], [DMy+kDMSize/2, DMy-kDMSize/2], 'k', linewidth=2)
plt.plot([OAP2_x, OAP2_x], [kAsgardHeight+kBeamWidth/2, kAsgardHeight-kBeamWidth/2], 'k', linewidth=2)

plt.text(kOAP1x-70, kAMBERHeight, 'OAP1')
plt.text(DMx+10, DMy, 'DM')
plt.text(OAP2_x-70, kAsgardHeight, 'OAP2')
    

plt.ylabel('Height above table (mm)')
plt.xlabel('Distance from AMBER table corner (mm)')
plt.tight_layout()
plt.savefig('../images/bc.png')