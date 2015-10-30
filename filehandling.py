import javabridge
import bioformats
from bioformats import log4j
import numpy as np

def start_java_bridge():
	javabridge.start_vm(class_path=bioformats.JARS)
	log4j.basic_config() 

def end_java_bridge():
	javabridge.kill_vm()

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

if __name__ == '__main__':
    main()