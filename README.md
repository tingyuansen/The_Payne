# The Payne
Tools for interpolating spectral models with neural networks. 

## Installation 
Clone this repository and add the base path to your PYTHONPATH variable. Or just run code from the base directory. 

Individual modules are imported separately, e.g:
```
import utils
import spectral_model
import fitting
```

The [tutorial](https://github.com/kareemelbadry/binspec/blob/master/tutorial.ipynb) shows some simple use cases. 

## Citing this code
* Please cite [Ting et al. 2018](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1804.01530), when using this code. The paper describes the method and its application to APOGEE spectra.
* If you use the routines relying on Jo Bovy's [apogee](https://github.com/jobovy/apogee) package to download spectra, please cite [Bovy (2016)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1510.06745).

## Authors
* [Yuan-Sen Ting](http://www.sns.ias.edu/~ting/) -- ting at ias dot edu
* [Kareem El-Badry](http://w.astro.berkeley.edu/~kelbadry/)

## Dependencies 
* The spectral model and fitting routines require only Numpy and Scipy.
* Routines to download and process APOGEE spectra require the [apogee](https://github.com/jobovy/apogee) package developed and maintained by Jo Bovy, and [Astropy](http://www.astropy.org/).
* Training a new neural network requires [PyTorch](http://pytorch.org/) (no GPUs required).
* I develop this package in Python 3 using Anaconda.

## Licensing

Copyright 2018 by Yuan-Sen Ting and Kareem El-Badry.

In brief, you can use, distribute, and change this package as you please. 

This software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
