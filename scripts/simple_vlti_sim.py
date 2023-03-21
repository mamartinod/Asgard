import numpy as np
import matplotlib.pyplot as plt
plt.ion()

x1 = np.array([4,3,2,1,2,1,0])
x2 = np.array([4,3,2,1,2,3,2])
x3 = np.array([4,3,4,5,6,5,4])
x4 = np.array([4,3,4,5,6,7,6])

bsx = np.array([3,1,5,2,6,1,3,5,7])
bsy = np.array([1,3,3,4,4,5,5,5,5])

y = np.arange(7)

xscale = 120
yscale = 50
yoffset = 350

plt.clf()
plt.plot(x1*xscale,yoffset-y*yscale)
plt.plot(x2*xscale,yoffset-y*yscale)
plt.plot(x3*xscale,yoffset-y*yscale)
plt.plot(x4*xscale,yoffset-y*yscale)
for x,y in zip(bsx,bsy):
    plt.plot([x * xscale,x * xscale], \
        [yoffset - (y+0.2)*yscale, yoffset - (y-0.2)*yscale], 'k', linewidth=2)
plt.xlabel('z (mm)')
plt.ylabel('x (mm)')

plt.tight_layout()

#Now also calculate throughput at 1.2 microns.
#Assuming ND0.3 splitters from Thorlabs:
R = 0.31
T = 0.56
efficiency = R**2 * T**2
print("Efficiency: {:.3f}".format(efficiency))