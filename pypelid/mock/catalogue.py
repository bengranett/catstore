import numpy as np
from scipy.spatial import cKDTree
import pypelid.sphere as sphere

class Catalogue(object):
    """ Base catalogue class. """
    ra = []
    dec = []
    properties = []

    def sub_catalogue(self, indices):
        """ Construct a new catalogue with the specified objects.

        Inputs
        ------
        indices - indices of objects

        Outputs
        -------
        catalogue
        """
        cat = type(self)()                        # construct new catalogue object

        # set attributes
        cat.ra = self.ra[indices]
        cat.dec = self.dec[indices]
        #cat.properties = self.properties[indices]
        return cat

    def build_tree(self):
        """ Initialize the data structure for fast spatial lookups. 

        Inputs
        ------
        None

        Outputs
        -------
        None
        """
        xyz = sphere.lonlat2xyz(self.ra, self.dec)
        self.lookup_tree = cKDTree(np.transpose(xyz))

    def query_cap(self, cra, cdec, radius=1.):
        """ Find neighbors to a given point (ra, dec). 

        Inputs
        ------
        cra - center ra degrees
        cdec - center dec degrees
        radius - degrees

        Outputs
        -------
        indices of objects in selection
        """
        r = radius * np.pi/180
        xyz = sphere.lonlat2xyz(cra,cdec)
        matches = self.lookup_tree.query_ball_point(np.transpose(xyz), r)
        return matches

    def query_box(self,  cra, cdec, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
        """ Find objects in a rectangle. 
        Inputs
        ------
        cra - center ra
        cdec - center dec
        width - width (degrees)
        height - height (degrees)
        padding - add this padding to width and height (degrees)
        orientation - (degrees about center)

        Outputs
        -------
        indices of objects in selection
        """
        r = np.sqrt(width**2 + height**2)/2.
        cap = self.query_cap(cra,cdec,r)

        ra = self.ra[cap]
        dec = self.dec[cap]

        dra,ddec = sphere.rotate_lonlat(ra, dec, [(orientation, cra, cdec)], inverse=True)

        sel_ra = np.abs(dra) < (width/2. + pad_ra)
        sel_dec = np.abs(ddec) < (height/2. + pad_dec)
        sel = np.where(sel_ra & sel_dec)
     
        matches = np.take(cap, sel[0])
        return matches

    def crop_box(self, cra, cdec, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
        """ Select objects in a rectangle and return a new catalogue object.

        Inputs
        ------
        cra - center ra
        cdec - center dec
        width - width (degrees)
        height - height (degrees)
        padding - add this padding to width and height (degrees)
        orientation - (degrees about center)

        Outputs
        -------
        catalogue
        """
        matches = self.query_box(cra, cdec, width, height, pad_ra, pad_dec, orientation)
        return self.sub_catalogue(matches)




