"""
Optical lock-in detection

This script reads a (x,y,t) .dv file and lets the user select
a region of interest from which a reference waveform is
calculated.This reference is used to calculate the time-
correlation pixel-by-pixel. The final output the correlation-
weighed image is stored as a .tiff file.
"""

import matplotlib.pyplot as pyplot
# from matplotlib.widgets import RectangleSelector
import numpy as np
from filehandling import *
from bioformats import omexml as ome

filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'


start_java_bridge()
image4d = readfile(filename)
imagexyt = image4d[:, :, 0, :]

freq = np.fft.fftn(imagexyt, axes=(0, 1, 2))
# freq = np.fft.fft2(imagexyt)


# print freqimage4d.shape
# print (freqimage4d).dtype

# print np.max(freq)
# print imagexyt.shape
freqslice = np.fft.fft2(freq[:, :, 1])
freqslice = np.fft.fftshift(freqslice)
writefile(filename[:-3]+'_FT2.tiff', np.abs(freqslice))
freq = np.fft.fftshift(freq)
freq = np.abs(freq)
writefile(filename[:-3]+'_FT.tiff', freq)
# writefile(filename[:-3] + '_FT.tiff', imagexyt)
# writefile('test10_FT.tiff', a/3)
end_java_bridge()

# " plotting "
# fig1 = pyplot.figure(1)
# ax1 = fig1.add_subplot(121)
# imgplot = ax1.imshow(image4d[:,:,0,0])
# ax2 = fig1.add_subplot(122)
# imgplot = ax2.imshow(np.sum(freqimage4d[:,:,:],2).astype(np.float64))
# pyplot.show()
