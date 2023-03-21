import numpy as np
import matplotlib.pyplot as plt
plt.ion()

input_sep = 240
output_sep = 125
ybif = -1000

xfold = -output_sep*np.arange(4)
yfold = -input_sep*np.arange(4)

plt.clf()
for i in range(4):
	plt.plot([-500, xfold[i], xfold[i]], [yfold[i], yfold[i], ybif-i*(input_sep+output_sep)], color='C'+str(i))
	x0 = xfold[i]
	y0 = ybif-i*(input_sep+output_sep)
	dy = input_sep+output_sep
	plt.plot([x0-output_sep+25,x0+225,x0+225,x0-output_sep+25,x0-output_sep+25], [y0, y0, y0-dy, y0-dy, y0], color='C'+str(i))
	print(y0)
	
plt.xlabel('x (mm)')
plt.tight_layout()