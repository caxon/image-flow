import numpy as np
from PIL import Image ,ImageDraw
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing
from scipy.ndimage import convolve
from skimage import measure
from kernels import *

default_zoom = (256, 256)

def chain(func):
	"""  method chaining decorator to return self"""
	def _chain(self, *args, **kwargs):
		func(self, *args, **kwargs)
		return self
	return _chain

# main class in image-flow
class Flow(object):
	""" Utility for chaining multiple preprocessing operations on an numpy array"""
	def __init__(self, array_or_filepath):
		if isinstance(array_or_filepath, np.ndarray) :
			img = np.array(array_or_filepath)
		elif isinstance(array_or_filepath, str):
			img = np.array(Image.open(array_or_filepath).convert("L"))
		else:
			raise Exception("Must start with np array or string")
		self.img_orig = img
		self.img = img
		self.labels = None
		self.regionprops = None
		self.regions = None

	@chain
	def conv(self, kernel=circular_kernel(25)):
		""" perform a convolution operation an image with an nxn kernel"""
		self.img = convolve(self.img, kernel, mode='constant' ,cval=0.0)

	@chain
	def norm(self):
		""" nomralize image from min..max to 0..1"""
		mins = np.min(self.img)
		maxs = np.max(self.img)
		ranges = maxs-mins
		self.img = (self.img- mins)/ ranges

	@chain
	def pos(self):
		""" Convert every entry to positive"""
		self.img= np.clip(self.img, a_min=0, a_max = None)

	@chain
	def thresh(self, thres_value=255):
		""" return a binary image 0 if below thresh_value else 1"""
		self.img= (self.img>=thres_value)*1.0

	@chain
	def flip(self, axis):
		""" mirror image across the middle of an axis. e.g. x mirror or y mirror"""
		self.img = np.flip(self.img, axis)

	@chain
	def transpose(self):
		""" transpose image"""
		self.img = np.transpose(self.img)

	@chain
	def thresh_scale(self, thres_value):
		""" Threshold data and scale by greyscale - threshold"""
		self.img= (self.img>=thres_value)*(self.img-thres_value+1.0)

	@chain
	def std(self, mean, std=1):
		""" Standardize data with mean and standard deviation"""
		self.img = (self.img - mean)/std

	@chain
	def dilation(self, iters=1, *, selem:np.ndarray=None):
		""" Binary dilation iters number of times. Selem is a binary mask representing neighborhood to consider"""
		for i in range(iters):
			self.img = binary_dilation(self.img, selem)*1.0

	@chain
	def erosion(self, iters =1, *, selem:np.ndarray	=None):
		""" Binary erosion iters number of times. Selem is a binary mask representing neighborhood to consider."""

		for i in range(iters):
			self.img = binary_erosion(self.img, selem)*1.0

	@chain
	def opening(self, iters=1,*,selem:np.ndarray=None):
		""" Erosion followed by dilation iters number of times. Selem is a binary mask representing neighborhood to consider."""

		for i in range(iters):
			self.img = binary_opening(self.img, selem)*1.0

	@chain
	def closing(self, iters=1,*,selem:np.ndarray	=None):
		""" Dilation followed by erosion iters number of times. Selem is a binary mask representing neighborhood to consider."""
		for i in range(iters):
			self.img = binary_closing(self.img, selem)*1.0

	@chain
	def reshape(self, shape):
		""" sets kernel to custom shape"""
		self.img = self.img.reshape(shape)

	@chain
	def label(self, *, connectivity=2):
		"""Label each disconnected region with a different (nonzero) color.

		**Updates self.labels feature**. Make this primary with Flow.focus("labels")"""
		self.labels = measure.label(self.img, connectivity=connectivity)

	@chain
	def highlight(self, xy, fill=None):
		""" put a dot at a single point on the image. xy is a tuple (x, y)"""
		x,y = xy
		drawer = ImageDraw.Draw(self.img)
		drawer.ellipse([x-2, y-2, x+2, y+2], fill=fill)

	@chain
	def highlights(self, xys, fill=None):
		""" put a dot a mutliple points on the image. xys is a list of tuples: [(x1,y1),...]"""
		for xy in xys:
			self.highlight(im, xy, fill)

	@chain
	def set_regionprops(self, *, connectivity=2):
		""" updates internal regionprops state. Does not modify img """
		self.labels = measure.label(self.img, connectivity=connectivity)
		self.regionprops = sorted(measure.regionprops(self.labels, coordinates='rc'), key=lambda x: x.bbox_area, reverse=True)
		if len( self.regionprops) < 3:
			print("regionprops is low: len={}".format(len(self.regionprops)))
		bboxes = []
		# test for overlapping bounding boxes (assuming 28x28 mnist source images)
		for i in range(len(self.regionprops)):
			row0, col0, row1, col1 = bbox = self.regionprops[i].bbox
			if (row1-row0>35 or col1-col0>35): # check if height or width are too large
				orientation = self.regionprops[i].orientation
				# if orientation is positive: digits lie in top-right and bottom-left
				if orientation >= 0:
					bboxes.append((row1-28, col0, row1, col0+28))
					bboxes.append((row0, col1-28, row0+28, col1))
				# if orientation is negative: digits lie in top-left and bottom-right
				else: # orientaiton < 0
					bboxes.append((row0, col0, row0+28, col0+28))
					bboxes.append((row1-28, col1-28, row1, col1))
			else:
				# bounding box probably contains one single digit
				bboxes.append(bbox)
		bboxes = bboxes[:3]
		if (len (bboxes) < 3): # later: extend by # required
			raise Exception("BBOXES SHOULD NEVER BE < 3")
		centers = [ [(b[2]+b[0])/2, (b[3]+b[1])/2 ] for b in bboxes ]
		self.regions = np.zeros((3,4), dtype=int)
		for row, center in enumerate(centers):
			if center[0] < 14: # check row (y) coordinates
				self.regions[row][0] = 0
				self.regions[row][2] = 28
			elif center[0] >=113:
				self.regions[row][0] = 100
				self.regions[row][2] = 128
			else:
				self.regions[row][0] = center[0] - 14
				self.regions[row][2] = center[0] + 14
			if center[1] < 14: # check column (x) coordinates
				self.regions[row][1] = 0
				self.regions[row][3] = 28
			elif center[1] >=113:
				self.regions[row][1] = 100
				self.regions[row][3] = 128
			else:
				self.regions[row][1] = center[1]-14
				self.regions[row][3] = center[1] + 14

	@chain
	def focus(self, attr_name):
		""" set self.img to attribute specified by string attr_name"""
		self.img = getattr(self, attr_name)

	@chain
	def unsequeeze(self):
		""" change image to 4d array (batch #, rows, cols, channels) for use in ML predicitons """
		self.img = np.reshape(self.img, (1,*self.img.shape,1))

	def extract(self, dil=0):
		# if idx is not None:
		# 	print(idx)
		""" get focal regions"""
		src_img = self.img #Flow(self.img_orig).thresh(230).norm().array()
		focal_regions = []
		for x0, y0, x1, y1 in self.regions:
			focal_regions.append(Flow(src_img[x0:x1, y0:y1]).dilation(dil).array())
		return np.stack(focal_regions,axis=0)

	def get_extractions(self):
		return self.extractions

	def predict(self, model):
		if self.regions is None:
			raise Exception("Must first compute focal regions with Flow.set_regionprops()")
		pds=  model.predict(self.extract(dil=1).reshape(3, 28, 28, 1))
		return np.argmax(pds, axis=1)

	def im(self):
		""" return PIL image"""
		return im(self.img)
	def imnz(self):
		""" returns PIL image on normalied array"""
		return imnz(self.img)
	def imz(self):
		""" returns zoomed PIL image """
		return imz(self.img)
	def imn(self):
		""" returns zoomed PIL image on normalied array"""
		return imn(self.img)

	def array (self):
		""" Get img ndarray"""
		return self.img

	def __repr__(self):
		return "Flow based on [{}*{}] np.ndarray.\nUse .im() to print an image and .get() to get ndarray".format(*self.img.shape)

# helper functions to create PIL image. Useful even outside of flow object.
def im(image):
	""" returns PIL image with floats from = 0...1"""
	if isinstance(image, np.ndarray):
		if np.max(image) > 1 or np.min(image < 0):
			raise Exception("Image np.ndarray must be normaized first.")
		return Image.fromarray(np.uint8(image*255), mode="L")
	else:
		raise Exception("Must import tensor, not ndarray or anything else")
def imn(image):
	""" returns PIL image with after being normalized"""
	return im(norm(image))
def imz(image, zoom = default_zoom):
	""" returns a zoomed image from ndarray with floats 0..1"""
	return im(image).resize(zoom)
def imnz(image, zoom=default_zoom):
	return imn(image).resize(zoom)

def norm(image):
	""" normalize image on range, putting max, min values in range 0,1"""
	if image is None:
		return None
	if isinstance(image, np.ndarray):
		mins = np.min(image)
		maxs = np.max(image)
	else:
		raise Exception(f"Got type: {type(image)}. Must be type ndarray")
	ranges = maxs-mins
	return (image- mins)/ranges