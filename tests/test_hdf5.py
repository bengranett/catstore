import sys
import numpy as np
import os
import tempfile
import pypelid.utils.hdf5tools as hdf5tools
import logging

logging.basicConfig(level=logging.DEBUG)


def test():
    """ """
    filename = tempfile.NamedTemporaryFile(delete=False).name+".pypelid"

    storage = hdf5tools.HDF5Catalogue(filename, 'w',
                preallocate_file=False)
    storage.update_attributes(attribute1='hi')
    storage.close()

    storage = hdf5tools.HDF5Catalogue(filename, 'a')
    storage.update_attributes(attribute2='ciao')
    storage.close()

    storage = hdf5tools.HDF5Catalogue(filename, 'r')
    assert storage.attribute1 == 'hi'
    assert storage.attribute2 == 'ciao'

    storage.show()

    storage.close()