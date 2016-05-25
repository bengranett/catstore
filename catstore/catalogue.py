import numpy as np
import fitsio
from sklearn.neighbors import KDTree
from pypelid.utils import sphere, misc
import healpy as hp

class Catalogue(object):
    """ Base catalogue class. """

    lon_name = 'alpha'
    lat_name = 'delta'

    data = None
    lookup_tree = None

    lon = None
    lat = None

    def __init__(self, cat=None):
        """
        Inputs
        ------
        cat - take data from this catalog if given.
        """
        if cat is not None:
            self.make_view(cat)
        else:
            self.data = np.array([])
        self.file = None

    def __getitem__(self, name):
        """ Return properties by name. """
        return self.data[name]

    def __str__(self):
        description = "Catalogue of " + str(len(self.data)) + " objects."
        return description

    def __len__(self):
        return len(self.data)

    def sub(self, indices):
        """ Construct a new catalogue with the specified index slice.

        Inputs
        ------
        slice - indices of objects

        Outputs
        -------
        catalogue
        """
        cat = Catalogue()

        # set attributes
        cat.data = self.data[indices]
        cat.lon = self.lon[indices]
        cat.lat = self.lat[indices]
        cat.lon_name = self.lon_name
        cat.lat_name = self.lat_name
        return cat

    def make_view(self, cat):
        """ Copy data in from a catalogue. """
        self.data = cat.data
        self.lookup_tree = cat.lookup_tree
        self.lon_name = cat.lon_name
        self.lat_name = cat.lat_name
        self.lon = cat.lon
        self.lat = cat.lat

    def read_cat(self, filename, names=None, converters=None, fits_ext=1):
        """ Read in a data file containing the input catalogue.

        Understands the following extensions:
        dat  - text file to be read with numpy.genfromtxt
        fits - fits file to be read with fitsio

        Inputs
        ------
        filename   - file to load
        names      - column names
        converters - dictionary with types to feed to numpy.genfromtxt (not used for fits)
        fits_ext   - fits extension number (default 1)

        Outputs
        -------
        structured array
        """

        if filename.endswith('.dat'):
            try:
                self.data = np.genfromtxt(filename, names=names, converters=converters)
            except ValueError as e:
                print e.message
                raise Exception("The input catalogue %s is in the wrong format. Run prepcat first!"%filename)

        elif filename.endswith('.fits'):
            try:
                self.data = fitsio.read(filename, columns=names, ext=fits_ext)
            except ValueError as e:
                print e.message
                raise Exception("The input catalogue %s is in the wrong format. Run prepcat first!"%filename)

        else:
            raise Exception("Unknown file type: %s"%filename)

        assert(self.data.dtype.fields.has_key(self.lon_name))
        assert(self.data.dtype.fields.has_key(self.lat_name))

        self.lon = self.data[self.lon_name].copy()  # copy because these coordinates can be transformed
        self.lat = self.data[self.lat_name].copy()

    def write_cat(self, filename):
        """ write a data file """
        raise Exception("Not implemented!")


    def build_tree(self):
        """ Initialize the data structure for fast spatial lookups.

        Inputs
        ------
        None

        Outputs
        -------
        None
        """
        xyz = sphere.lonlat2xyz(self.lon, self.lat)
        self.lookup_tree = KDTree(np.transpose(xyz))

    def project(self, transform):
        """ Apply spatial projection. """
        self.lon, self.lat = transform(self.lon, self.lat)

    def to_cartesian(self):
        """ Return a cartesian catalogue """
        return CartesianCatalogue(self)

    def query_cap(self, clon, clat, radius=1.):
        """ Find neighbors to a given point (clon, clat).

        Inputs
        ------
        clon - center longitude degrees
        clat - center latitude degrees
        radius - degrees

        Outputs
        -------
        indices of objects in selection
        """
        if self.lookup_tree is None: self.build_tree()

        r = radius * np.pi/180

        xyz = sphere.lonlat2xyz(clon, clat)

        if misc.is_number(xyz[0]):
            xyz = np.transpose(xyz).reshape(1,-1)
        else:
            xyz = np.transpose(xyz)

        matches = self.lookup_tree.query_radius(xyz, r)
        return matches

    def query_box(self,  clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
        """ Find objects in a rectangle.
        Inputs
        ------
        clon - center longitude
        clat - center latitude
        width - width (degrees)
        height - height (degrees)
        padding - add this padding to width and height (degrees)
        orientation - (degrees about center)

        Outputs
        -------
        indices of objects in selection
        """
        r = np.sqrt(width**2 + height**2)/2.
        cap = self.query_cap(clon,clat,r)[0]

        lon = self.lon[cap]
        lat = self.lat[cap]

        dlon,dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)], inverse=True)

        sel_lon = np.abs(dlon) < (width/2. + pad_ra)
        sel_lat = np.abs(dlat) < (height/2. + pad_dec)
        sel = np.where(sel_lon & sel_lat)

        matches = np.take(cap, sel[0])
        return matches

    def crop_box(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
        """ Select objects in a rectangle and return a new catalogue object.

        Inputs
        ------
        clon - center ra
        clat - center dec
        width - width (degrees)
        height - height (degrees)
        padding - add this padding to width and height (degrees)
        orientation - (degrees about center)

        Outputs
        -------
        catalogue
        """
        matches = self.query_box(clon, clat, width, height, pad_ra, pad_dec, orientation)
        return self.sub(matches)

    def plot(self):
        """ Create a Mollweide projected plot of the objects.

        Inputs
        ------
        None

        Outputs
        ------
        None
        """

        # Coordinates in radians
        x = misc.torad(self.lon)
        y = np.pi-(misc.torad(self.lat)+np.pi/2.0)

        # Setup healpix
        fig = hp.visufunc.projscatter(y, x, s=10., lw=0.0)
        hp.graticule()

        return fig

class CartesianCatalogue(Catalogue):
    """ """
    def build_tree(self):
        """ Initialize the data structure for fast spatial lookups.

        Inputs
        ------
        None

        Outputs
        -------
        None
        """
        self.lookup_tree = KDTree(np.transpose([self.lon, self.lat]))

    def query_disk(self, x, y, radius=1.):
        """ Find neighbors to a given point (ra, dec).

        Inputs
        ------
        cx - center x
        cy - center y
        radius - radius

        Outputs
        -------
        indices of objects in selection
        """
        if self.lookup_tree is None: self.build_tree()

        matches = self.lookup_tree.query_radius(np.transpose([x,y]), radius)
        return matches

    query_cap = query_disk

    def query_box(self,  cx, cy, width=1, height=1, pad_x=0.0, pad_y=0.0, orientation=0.):
        """ Find objects in a rectangle.
        Inputs
        ------
        cx - array center x
        cy - array center y
        width - float width
        height - float height
        pad_x - float add this padding to width
        pad_y - float add this padding to height
        orientation - float position angle of the box

        Outputs
        -------
        list of indices of objects in selection
        """
        try:
            len(cx)
        except TypeError:
            cx = np.array([cx])
            cy = np.array([cy])
        theta = misc.torad(orientation)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        r = np.sqrt(width**2 + height**2)/2.

        matches = self.query_disk(cx,cy,r)

        results = []
        for i, match in enumerate(matches):
            dx = self.lon[match] - cx[i]
            dy = self.lat[match] - cy[i]
            dxt = dx * costheta - dy * sintheta
            dyt = dx * sintheta + dy * costheta

            sel_x = np.abs(dxt) < (width/2. + pad_x)
            sel_y = np.abs(dyt) < (height/2. + pad_y)
            sel = np.where(sel_x & sel_y)

            results.append(np.take(match, sel[0]))
        if len(results)==1:
            return results[0]
        return results
