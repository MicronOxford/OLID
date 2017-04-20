"""
Optical lock-in detection

This script reads a (x,y,t) .dv file and lets the user select
a region of interest from which a reference waveform is
calculated.This reference is used to calculate the time-
correlation pixel-by-pixel. The final output the correlation-
weighed image is stored as a .tiff file.
"""

import matplotlib.pyplot as pyplot
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
from filehandling import *

filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
# # filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'
# filename = 'data/C127_pKer_RSEGF2_1ms10pct405nm_4ms10pct488nm_20151023_Fri-1630.dv'
# filename = 'data/C127_pKer_RSEGF2_1ms10pct405nm_5ms10pct488nm_20151023_Fri-1558.dv'
# filename = 'data/2015-09-A-C127_VimN205S_pre_3_R3D.dv'
# filename = 'data/04_20x_1ms10pc405_20x2ms1pct488_20151119_Thu-1907.dv'

start_java_bridge()
image4d, metaxml = readfile(filename)
# imagexyt = image4d[:, :, 0, :]

Pixels = metaxml.image().Pixels
nx, ny, nz, nt = Pixels.SizeX, Pixels.SizeY, Pixels.SizeZ, Pixels.SizeT
imagexyt = image4d.reshape((nx, ny, nz*nt), order='F')
imagexyt = imagexyt[:, :, :]

# 3D Fourier Transform
freq = np.fft.fftn(imagexyt, axes=(0, 1, 2))

(nx, ny, nt) = freq.shape
freqslice = np.zeros((nx, ny, nt), dtype=np.complex)

for i in range(0, nt):
    freqslice[:, :, i] = np.fft.ifft2(freq[:, :, i])


exptime = metaxml.image().Pixels.Plane(0).ExposureTime
if exptime == 0:
    exptime = 1.0/nt


" plotting "
# prepare for plotting
freqdisp = np.fft.fftshift(freq)
freqdisp = np.abs(freqdisp)
freqdisp = np.sum(freqdisp, axis=0)
freqdisp = np.log(freqdisp)

fig1 = pyplot.figure(1)
ax3 = fig1.add_axes([0.9, 0.05, 0.09, 0.05])  # save button
ax4 = fig1.add_axes([0.8, 0.05, 0.09, 0.05])  # save button
ax1 = fig1.add_subplot(121)
ax1.data = freqdisp
ax1.nt = nt
ax1.frequencies = np.round(np.fft.fftfreq(nt, d=exptime), decimals=5)
ax2 = fig1.add_subplot(122)
ax2.slice = [0]
# ax2.slice = [0, 1]
ax2.data = freqslice
ax2.nt = nt
ax2.frequencies = ax1.frequencies
ax2.slice = np.asarray(ax2.slice)


def plot_freq(ax):
    ax.clear()
    ax.imshow((ax.data).transpose(), cmap='hot')
    ax.set_title('3D Fourier Transform (x-projection)')
    ax.set_xlabel('y [px]')
    ax.set_ylabel('time frequency [Hz]')
    if np.remainder(ax.nt, 2) == 0:
        fmaxidx = ax.nt/2-1
    else:
        fmaxidx = (ax.nt-1)/2
    label = np.hstack((ax.frequencies[fmaxidx+1:],
                       ax.frequencies[0:fmaxidx+1]))
    labelstep = ax.nt/(len(ax.get_yticks())-1-1)
    yticklabel = np.hstack((np.array(['']), label[::labelstep].astype(str)))
    ax.set_yticklabels(yticklabel)
    pyplot.draw()


def plot_freqslice(ax):
    idx = np.fft.fftshift(range(0, ax.nt))
    fmin = -np.ceil((ax.nt-1)/2.0)
    sliceidx = idx[(-fmin+ax.slice).tolist()]
    freqslicedisp = np.squeeze(np.abs(ax.data[:, :, sliceidx]))
    if np.ndim(freqslicedisp) > 2:
        freqslicedisp = np.sum(freqslicedisp, axis=2)
        title = 'Slice frequency:  %.1f' % ax.frequencies[sliceidx[0]]
        title += ' to %.1f Hz' % ax.frequencies[sliceidx[-1]]
    else:
        title = 'Slice frequency: %.1f Hz' % ax.frequencies[sliceidx]
    ax.clear()
    ax.imshow(freqslicedisp, cmap='hot')
    ax.set_title(title)
    pyplot.draw()


def onscroll(event):
    ax = fig1.get_axes()[-1]
    if np.size(ax.slice) is not 1:
        return
    fmin = -np.ceil((ax.nt-1)/2.0)
    fmax = np.floor((ax.nt-1)/2.0)
    if (ax.slice + event.step) < fmin:
        if ax.slice == fmin:
            return
        else:
            ax.slice[0] = fmin
    elif fmax < (ax.slice + event.step):
        if ax.slice == fmax:
            return
        else:
            ax.slice[0] = fmax
    else:
        ax.slice += event.step
    plot_freqslice(ax)


def onselect(eclick, erelease):
    ystart, yend = int(eclick.ydata), int(erelease.ydata)
    # if ystart is yend:
    #     return
    if ystart > yend:
        ystart, yend = yend, ystart
    ax = fig1.get_axes()[-1]
    fmin = -np.ceil((ax.nt-1)/2.0)
    ax.slice = np.arange(ystart, yend+1) + fmin
    plot_freqslice(ax)


def save_img(event):
    ax = fig1.get_axes()[-1]
    idx = np.fft.fftshift(range(0, ax.nt))
    fmin = -np.ceil((ax.nt-1)/2.0)
    sliceidx = idx[(-fmin+ax.slice).tolist()]
    freqslicedisp = np.squeeze(np.abs(ax.data[:, :, sliceidx]))
    if ax.slice.size > 1:
        print 'Multiple slice saving not implemented yet.'
        print 'Please select single slice.'
        return
    savename = filename[:-3]
    savename += '_%05.dmHz.tiff' % (1000*ax.frequencies[sliceidx])
    writefile(savename, freqslicedisp)
    print 'Frequency %.1f Hz saved as %s' % (1000*ax.frequencies[sliceidx], savename)


def save_fftstack(event):
    ax = fig1.get_axes()[-1]
    data = np.abs(ax.data)
    savename = filename[:-3] + '_fftslices.tiff'
    writefile(savename, data, t=3, size_t=ax.nt)
    print 'Saved stack to %s' % savename


plot_freq(ax1)
plot_freqslice(ax2)
cid = fig1.canvas.mpl_connect('scroll_event', onscroll)
lineprops = dict(color='green', linestyle='-', linewidth=5, alpha=0.5)
RS = RectangleSelector(ax1, onselect,  drawtype='line', lineprops=lineprops)


save_ax2 = Button(ax3, 'save slice')
save_ax2.on_clicked(save_img)
save_fftstack_button = Button(ax4, 'save stack')
save_fftstack_button.on_clicked(save_fftstack)

pyplot.show()
end_java_bridge()
