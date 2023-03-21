import numpy as np
import matplotlib.pyplot as plt
plt.ion()

oap1_ROC = 1363.12
oap1_decentre = 44.8
oap1_rad = 25.4
oap1_angle = np.radians(3.765)

oap2_ROC = 903.669
oap2_decentre = np.sqrt(67**2 + 29.6**2)
oap2_rad = 12.7
oap2_angle = np.radians(9.268)

centre_thickness = 15

for ix, (ROC, decentre, rad, angle) in enumerate(zip([oap1_ROC,oap2_ROC],\
	[oap1_decentre,oap2_decentre], [oap1_rad, oap2_rad],[oap1_angle, oap2_angle])):
	plt.figure(ix+1)
	plt.clf()
	x = np.linspace(decentre-rad, decentre+rad,100)
	yoffset = 1/(2*ROC) * decentre**2
	y = 1/(2*ROC) * x**2  - yoffset + centre_thickness
	plt.plot(x, y,'b')
	plt.plot([x[0],x[0],x[-1],x[-1]],[y[0],0,0,y[-1]],'b')
	ylin = y[0] + (y[-1]-y[0])/(x[-1]-x[0])*(x - x[0])
	plt.plot(x, ylin, 'g')
	print("Maximum cutting depth: {:.2f}mm".format(np.max(ylin-y)))
	plt.xlabel('Radius (mm)')
	plt.ylabel('Height (mm)')
	plt.tight_layout()
	plt.savefig('OAP{:d}.png'.format(ix+1))