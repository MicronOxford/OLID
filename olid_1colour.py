'''
Optical lock-in detection

This script reads a (x,y,t) .dv file and lets the user select a region of interest from which a reference waveform is calculated. This reference is used to calculate the time-correlation pixel-by-pixel. The final output the correlation-weighed image is stored as a .tiff file.
'''
import matplotlib.pyplot as pyplot
from matplotlib.widgets import RectangleSelector
import pylab
import numpy as np
import time
import javabridge
import bioformats
from bioformats import log4j

# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'
corr_threshold = 0.7
timepts = [] # set time points used for correlation, use all if empty

def readfile(filename):
	# read metadata
	metadata = bioformats.get_omexml_metadata(filename)
	xml = bioformats.OMEXML(metadata)
	Pixels = xml.image().Pixels
	nx, ny, nz, nt = Pixels.SizeX, Pixels.SizeY, Pixels.SizeZ, Pixels.SizeT

	# read image data
	image4d = np.zeros(shape=(nx,ny,nz,nt))
	reader = bioformats.ImageReader(filename)
	for t in range(nt):
		for z in range(nz):
			image4d[:,:,z,t] = reader.read(z=z, t=t, rescale=False)

	return image4d

def writefile(filename, image):
	# write image data
	bioformats.write_image(filename, image, pixel_type=bioformats.omexml.PT_UINT16)

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

javabridge.start_vm(class_path=bioformats.JARS)
log4j.basic_config() 
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
javabridge.kill_vm()


'''plotting'''
fig = pyplot.figure(1, figsize=(12,6))

# plot sum image
ax1 = pyplot.subplot(141)
imgplot = ax1.imshow(np.reshape(np.mean(Ipxtimeseries,axis=1), (nx,ny)), interpolation='none')
imgplot.set_cmap('gray')
pyplot.xticks([])
pyplot.yticks([])

# plot selection
ax2 = pyplot.subplot(142)
imgplot = ax2.imshow(np.reshape(np.mean(REFpxtimeseries,axis=1), (RS.dims[0],RS.dims[1])), interpolation='none')
pyplot.xticks([])
pyplot.yticks([])

# ax3 = pyplot.subplot(143, aspect='equal')
pyplot.rc('xtick', labelsize='6', direction='in')
pyplot.rc('ytick', labelsize='6', direction='in')
ax3 = fig.add_subplot(143)
REFplot = ax3.plot(REF)
ax3.set_aspect(1./ax3.get_data_ratio())

# plot corr
ax4 = pyplot.subplot(144)
corrplot = ax4.imshow(Icorr, interpolation='none')
corrplot.set_cmap('gray')
pyplot.xticks([])
pyplot.yticks([])

pyplot.show()