import sys
import numpy as np
import pylab
import pypelid.sky.catalogue_store as catalogue_store

import logging
logging.basicConfig(level=logging.DEBUG)


def load(filename):
    """ """
    pylab.subplot(aspect='equal')

    with catalogue_store.CatalogueStore(filename) as cat:
        cat.plot()

    pylab.show()

if __name__=="__main__":
    load(sys.argv[1])