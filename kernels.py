import numpy as np

# generic useful kernels with minimal dependencies
def circular_kernel(size, binary=False):
	""" Returns a circular kernel np.array of size x size dims"""
	if size % 2== 0 or size <1:
		raise Exception("Size must be positive and odd")
	kernel = np.zeros((size, size))
	for n in range(size**2):
		x,y = n%size, n//size
		dist = np.hypot(x- size//2,y-size//2)
		kernel[x,y]= dist
	kernel = 1-kernel/np.sqrt(2*(size//2) **2)
	if binary:
		return (kernel > 0.2)*1.0
	return kernel

def g_kernel(size, sig=1.):
	""" creates a gaussian kernel with of dims : size x size and sigma = sig."""
	if size % 2== 0 or size <1:
		raise Exception("Size must be positive and odd")
	ax = np.linspace(-(size-1)/2., (size-1)/2., size).astype(np.float32)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
	return kernel/np.sum(kernel)

def id_kernel(size):
	""" return an id kernel with a single 1 in the middle"""
	if size % 2== 0 or size <1:
		raise Exception("Size must be positive and odd")
	kernel = np.zeros((size, size))
	kernel[size//2,size//2] =1


#custom useful kernels
c5= circular_kernel(5)
c11= circular_kernel(11)
c15= circular_kernel(15)
c5f= circular_kernel(5, binary=True)
c11f= circular_kernel(11,binary=True)
c15f= circular_kernel(15, binary=True)
b3 = np.array([
	[1,1,1],
	[1,1,1],
	[1,1,1]
])
d5 = np.array([
	[0,0,1,0,0],
	[0,1,1,1,0],
	[1,1,1,1,1],
	[0,1,1,1,0],
	[0,0,1,0,0]
])
cr5 = np.array([
	[0,0,1,0,0],
	[0,0,1,0,0],
	[1,1,1,1,1],
	[0,0,1,0,0],
	[0,0,1,0,0]
])
