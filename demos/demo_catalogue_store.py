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

        for dec in range(0,10,2):
            stuff = cat.retrieve(45.,dec,width=2.,height=2.,orientation=0)

            x,y = np.transpose(stuff['skycoord'])
            pylab.plot(x,y,".")


    with catalogue_store.CatalogueStore(filename) as cat:
        cat.plot()

    pylab.show()

if __name__=="__main__":
    load(sys.argv[1])