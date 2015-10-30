'''
Optical lock-in detection

This script reads a (x,y,t) .dv file and lets the user select a region of interest from which a reference waveform is calculated. This reference is used to calculate the time-correlation pixel-by-pixel. The final output the correlation-weighed image is stored as a .tiff file.
'''
import matplotlib.pyplot as pyplot
from matplotlib.widgets import RectangleSelector
import numpy as np
from filehandling import *

# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'
corr_threshold = 0.7
timepts = [] # set time points used for correlation, use all if empty



def onselect(eclick, erelease):
	'eclick and erelease are matplotlib events at press and release'
	if eclick.xdata < erelease.xdata:
		xstart, xend = eclick.xdata, erelease.xdata
	elif eclick.xdata > erelease.xdata:
		xend, xstart = eclick.xdata, erelease.xdata

	if eclick.ydata < erelease.ydata:
		ystart, yend = eclick.ydata, erelease.ydata
	elif eclick.ydata > erelease.ydata:
		yend, ystart = eclick.ydata, erelease.ydata	

	try:
		RS.coords = np.array([ystart, xstart, yend, xend]) # image and plot reverse axes
		pyplot.close()
	except:
		return

start_java_bridge()
image4d = readfile(filename)

[nx, ny, nz, nt] = image4d.shape
if timepts == []:
	timepts = range(0,nt) # set time points used for correlation


''' select area used as reference waveform '''
fig = pyplot.figure(2)
ax = pyplot.gca()
REFimg = ax.imshow(np.mean(image4d[:,:,0,:],axis=2), interpolation='none')
RS = RectangleSelector(ax, onselect)
RS.coords = np.array([0,0,ny-1,nx-1]) # image and plot reverse axes
pyplot.show()

RS.coords = RS.coords.astype(int)
RS.dims = np.array([RS.coords[2]-RS.coords[0], RS.coords[3]-RS.coords[1]])

# generate signal wave
Ipxtimeseries = np.reshape(image4d[:,:,0,:], (nx*ny,nt))
I = Ipxtimeseries[:,timepts]
mI = np.mean(I, axis=1)
sI = np.std(I, axis=1)

# generate reference wave
REFimg = image4d[RS.coords[0]:RS.coords[2],RS.coords[1]:RS.coords[3],0,:]
REFpxtimeseries = np.reshape(REFimg, (RS.dims[0]*RS.dims[1],nt))
# REF = np.percentile(REFpxtimeseries[:,timepts], 90, axis=0)
REF = np.mean(REFpxtimeseries[:,timepts], axis=0)
mREF = np.mean(REF, axis=0)
sREF = np.std(REF, axis=0)

''' calculate correlation corr(x,y) '''
corr = np.zeros(I.shape[0])
for t in range(I.shape[1]):
	corr += (I[:,t]-mI)*(REF[t]-mREF)/(sI*sREF)
corr /= I.shape[1]
corr[corr<corr_threshold] = 0

Icorr = np.reshape(np.mean(Ipxtimeseries,axis=1)*corr, (nx,ny))

writefile(filename[:-3]+'.tiff', Icorr)
end_java_bridge()


'''plotting'''
fig = pyplot.figure(1, figsize=(12,3), frameon=False)
fig.width = 12.0
fig.height = 3.0

ncols = 4 # number of subplots
nrows = 1
spacing = 0.15
spacingx = spacing / ncols
spacingy = spacing / nrows
lengthx = (1.0-(ncols+1)*spacingx)/ncols 
lengthy = (1.0-(nrows+1)*spacingy)/nrows

# plot sum image
x1 = 0+spacingx
y1 = 0+spacingy
ax1 = fig.add_axes([x1, y1, lengthx, lengthy])
imgplot = ax1.imshow(np.reshape(np.mean(Ipxtimeseries,axis=1), (nx,ny)), interpolation='none')
imgplot.set_cmap('gray')
pyplot.xticks([])
pyplot.yticks([])

# plot selection
x2 = (spacingx+lengthx) + x1
y2 = y1
ax2 = fig.add_axes([x2, y2, lengthx, lengthy])
imgplot = ax2.imshow(np.reshape(np.mean(REFpxtimeseries,axis=1), (RS.dims[0],RS.dims[1])), interpolation='none')
pyplot.xticks([])
pyplot.yticks([])

# plot ref waveform
pyplot.rc('xtick', labelsize='6', direction='in')
pyplot.rc('ytick', labelsize='6', direction='in')
x3 = (spacingx+lengthx) + x2
y3 = y2
ax3 = fig.add_axes([x3, y3, lengthx, lengthy])
REFplot = ax3.plot(REF)
ax3.set_aspect(1./ax3.get_data_ratio())

# plot corr
x4 = (spacingx+lengthx) + x3
y4 = y3
ax4 = fig.add_axes([x4, y4, lengthx, lengthy])
corrplot = ax4.imshow(Icorr, interpolation='none')
corrplot.set_cmap('gray')
pyplot.xticks([])
pyplot.yticks([])

def onresize(event):
	width = fig.get_figwidth() -0.125
	height = fig.get_figheight() -0.125
	if width/height < fig.width/fig.height:
		# fig.set_figwidth(height*fig.width/fig.height+0.00001)
		fig.set_figheight(width*fig.height/fig.width-0.00001)
	# ax = fig.get_axes()

cid = fig.canvas.mpl_connect('resize_event', onresize)

pyplot.show()