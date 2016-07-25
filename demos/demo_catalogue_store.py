import sys
import numpy as np
import pylab
import pypelid.sky.catalogue_store as catalogue_store

import logging
logging.basicConfig(level=logging.DEBUG)


def load(filename):
    """ """
    cat = catalogue_store.CatalogueStore(filename)
    
    stuff = cat.retrieve(45.,0.,width=20.,height=10.,orientation=20)

    x,y = np.transpose(stuff['skycoord'])
    pylab.plot(x,y,".")
    pylab.show()

if __name__=="__main__":
    load(sys.argv[1])