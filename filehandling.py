import javabridge
import bioformats
from bioformats import log4j
import numpy as np
from bioformats import omexml as ome
import os


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
    image4d = np.zeros(shape=(nx, ny, nz, nt))
    reader = bioformats.ImageReader(filename)
    for t in range(nt):
        for z in range(nz):
            image4d[:, :, z, t] = reader.read(z=z, t=t, rescale=False)

    return image4d, xml


def writefile(filename,
              image,
              c=0,
              z=0,
              t=0,
              size_c=1,
              size_z=1,
              size_t=1,
              channel_names=None):
    # write image data
    # if file exists already, we have to remove it to prevent java error of
    # unmatching pixel dimensions
    if os.path.isfile(filename):
        os.remove(filename)
    bioformats.write_image(filename,
                           image.astype(ome.PT_FLOAT),
                           pixel_type=ome.PT_FLOAT,
                           c=c,
                           z=z,
                           t=t,
                           size_c=size_c,
                           size_z=size_z,
                           size_t=size_t,
                           channel_names=channel_names)
