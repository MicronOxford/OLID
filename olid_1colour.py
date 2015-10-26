import matplotlib.pyplot as pyplot
import numpy as np
import time
import javabridge
import bioformats
from bioformats import log4j

def readfile(filename):
	# read metadata
	metadata = bioformats.get_omexml_metadata(filename)
	xml = bioformats.OMEXML(metadata)
	Pixels = xml.image().Pixels
	nx = Pixels.SizeX
	ny = Pixels.SizeY
	nz = Pixels.SizeZ
	nt = Pixels.SizeT

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

javabridge.start_vm(class_path=bioformats.JARS)
log4j.basic_config() 


# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_3_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_6_R3D.dv'
# filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_10_R3D.dv'
filename = 'data/2015-09-A-C127_VimN205S_post20min_2x50nM_9_R3D.dv'
corr_threshold = 0.7


image4d = readfile(filename)
[nx, ny, nz, nt] = image4d.shape

timepts = range(0,nt) # set time points used for correlation


# generate reference wave
pxtimeseries = np.reshape(image4d[:,:,0,:], (nx*ny,nt))
REF = np.percentile(pxtimeseries[:,timepts], 90, axis=0)
pyplot.figure(1)
REFplot = pyplot.plot(REF)
pyplot.draw()

# calculate correlation coefficient per pixel
mREF = np.mean(REF, axis=0)
sREF = np.std(REF, axis=0)

I = pxtimeseries[:,timepts]
mI = np.mean(I, axis=1)
sI = np.std(I, axis=1)

# calculate correlation corr(x,y)
corr = np.zeros(I.shape[0])
for t in range(I.shape[1]):
	corr += (I[:,t]-mI)*(REF[t]-mREF)/(sI*sREF)
corr /= I.shape[1]
corr[corr<corr_threshold] = 0

Icorr = np.reshape(np.mean(pxtimeseries,axis=1)*corr, (nx,ny))


pyplot.figure(2)
# plot sum image
pyplot.subplot(121)
imgplot = pyplot.imshow(np.reshape(np.mean(pxtimeseries,axis=1), (nx,ny)), interpolation='none')
imgplot.set_cmap('gray')

# plot corr
pyplot.subplot(122)
corrplot = pyplot.imshow(Icorr, interpolation='none')
corrplot.set_cmap('gray')
pyplot.draw()


writefile(filename[:-3]+'.tiff', Icorr)

javabridge.kill_vm()
pyplot.show()