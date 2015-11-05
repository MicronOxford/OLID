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

# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'


start_java_bridge()
image4d = readfile(filename)
imagexyt = image4d[:, :, 0, :]

# 3D Fourier Transform
freq = np.fft.fftn(imagexyt, axes=(0, 1, 2))

# print np.max(freq)
# print imagexyt.shape

(nx, ny, nt) = freq.shape
freqslice = np.zeros((nx, ny, nt), dtype=np.complex)

for i in range(0, nt):
    freqslice[:, :, i] = np.fft.ifft2(freq[:, :, i])

# freqslice = np.fft.fftshift(freqslice)
# writefile(filename[:-3]+'_FT2.tiff', np.abs(freqslice))
# freq = np.fft.fftshift(freq)
# freq = np.abs(freq)
# writefile(filename[:-3]+'_FT.tiff', freq)

# writefile(filename[:-3] + '_FT.tiff', imagexyt)
# writefile('test10_FT.tiff', a/3)
end_java_bridge()

" plotting "
# prepare for plotting
freqdisp = np.fft.fftshift(freq)
freqdisp = np.abs(freqdisp)
freqdisp = np.sum(freqdisp, axis=0)
freqdisp = np.log(freqdisp)
# freqslicedisp = np.fft.ifftshift(freqslice)
freqslicedisp = np.abs(freqslice[:, :, 0])


fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(121)
ax1.data = freqdisp
ax1.nt = nt
# ax1.set_ylabel('time')
ax2 = fig1.add_subplot(122)
ax2.slice = 0
ax2.data = freqslice
ax2.nt = nt


def plot_freq(ax):
    ax.clear()
    ax.imshow((ax.data).transpose(), cmap='hot')
    ax.set_title('3D Fourier Transform (x-projection)')
    ax.set_xlabel('y')
    ax.set_ylabel('time frequency')
    fmax = np.floor((ax.nt-1)/2.0)
    print ax.get_yticks()
    # idx_label = np.arange(fmin, fmax, 5).astype(str).tolist()
    n = 20
    idx_label = np.arange(0, fmax, n)
    idx_label = np.append(-idx_label[-1:0:-1], idx_label).astype(str).tolist()
    # idx = np.arange(0, ax.nt, 5).tolist()
    step = ax.nt/len(idx_label)
    idx = np.arange(step/2, ax.nt, step).tolist()
    print idx
    # print idx.astype(str)
    ax.set_yticks(idx)
    ax.set_yticklabels(idx_label)
    # ax.set_yticks([0, 1.4, 88])
    # # print idx[::10]
    # pyplot.draw()


def plot_freqslice(ax):
    idx = np.fft.fftshift(range(0, ax.nt))
    fmin = -np.ceil((ax.nt-1)/2.0)
    sliceidx = idx[-fmin+ax.slice]
    freqslicedisp = np.abs(ax.data[:, :, sliceidx])
    ax.clear()
    ax.imshow(freqslicedisp, cmap='hot')
    ax.set_title('Slice frequency: %.3f' % np.fft.fftfreq(ax.nt)[sliceidx])
    pyplot.draw()


def onscroll(event):
    ax = fig1.get_axes()[-1]
    fmin = -np.ceil((ax.nt-1)/2.0)
    fmax = np.floor((ax.nt-1)/2.0)
    if (ax.slice + event.step) < fmin:
        if ax.slice == fmin:
            return
        else:
            ax.slice = fmin
    elif fmax < (ax.slice + event.step):
        if ax.slice == fmax:
            return
        else:
            ax.slice = fmax
    else:
        ax.slice += event.step
    plot_freqslice(ax)

plot_freq(ax1)
plot_freqslice(ax2)
cid = fig1.canvas.mpl_connect('scroll_event', onscroll)


pyplot.show()
