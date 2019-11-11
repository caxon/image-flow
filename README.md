# image-flow
## About
Python image modification class using builder/method chaining design pattern. Originally intended for image preprocessing for ML models. Incorperates a suite of image processing operations from numpy, skimage, scipy image, and PIL.

### Advantanges of image-flow
 - Easy to perform multiple consecutive operations on an image
 - Easy to visualize image (with im/imnz/rgb functions) in iPython for pipeline development
 
### Special use-cases:
 - Find highlights (e.g. MNIST digits) on a busy background
 
## How to use:
1. Construct a Flow object with the input as source:

```Flow(numpy_array_2d)``` or ```Flow("path/to/image.png")```

2. Call consecutive image operations on the returned product. Each image operation returns the modified flow object, designed to make chaining function calls easier

Ex: to find the binary threshold of an image, dilate the image twice, and flip along the x axis: 
```Flow(np_array).thresh(150).dilate(2).flip(axis=0)```

3. Return the image as a PIL gresycale image:

```Flow(np_arrray).im()```

## Functions supported: 
* **flip(axis:int):** flip image along axis.
* **thresh(thresh_value):** threshold image at a certain value. I.e. all entries > value return 1, all entries < value return 0.
* **thresh_scale(thres_value):** all pixels below the value become 0, all pixels above the value are scaled from 0 to original value.
* **std(mean, std_deviation):** standarize image with mean and std_deviation: (pixel[i] - mean) / std_deviation
* **pos():** convert every entry to positive (>= 0)
* **erosion, dilation, closing, opening:** perform the binary morphological operation. Optional arg itters determines how many times to repeat operation. Optional arg selem specifies a binary nxn np.ndarray which specifies neighbors considered in funciton.

### Returning functions:
These functions end the chain of methdods and return a value
* **im():** returns a PIL image (data must range from 0->1)
* **imn():** returns a normalized PIL image
* **imz():** returns a zoomed (256x256 pixesl by defualt) and *unnormalized* PIL image
* **imnz():** returns a zoomed and normalized PIL image
* **array():** returns the raw data in the array

### Todo list:
 - Implement parallelization using [joblib](https://joblib.readthedocs.io/en/latest/parallel.html).
 - Display multiple images on the same line using matplotlib.pyplot to draw images
 - Implement additional greyscale operations
 - Improve support for multichannel operations
 
