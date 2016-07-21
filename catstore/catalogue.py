import logging
import numpy as np
import h5py
import fitsio
from sklearn.neighbors import KDTree
from pypelid.utils import sphere, misc

class Catalogue(object):
    """ Base catalogue class. """

    _lon_name = 'alpha'
    _lat_name = 'delta'

    def __init__(self, cat=None, data=None, filename=None, 
        zone_resolution=5, hp_order='nest'):
        """
        Inputs
        ------
        cat - take data from this catalog if given.
        """
        if cat is not None:
            self.make_view(cat)
            return

        self._filename = filename

        if os.path.exists(filename):
            self.read(filename)

        self._lookup_tree = None
        self._file = None
        self._index = None

    def __getattr__(self, key):
        """ Return columns by name 

        Columns may be accessed by catalogue.alpha
        """
        # if the attribute is a class variable, return it
        try:
            return self.__dict__[key]
        except KeyError:
            pass
        # if the attribute is a column name in the data table, return the column
        try:
            return self.__dict__['_data'][key]
        except KeyError:
            raise AttributeError
        except IndexError:
            raise AttributeError
        except ValueError:
            raise AttributeError

    def __setattr__(self, key, value):
        """ Set a catalogue attribute.  

        Raises CatalogueError exception if the variable name exists in the data table. 
        """

        if self.__dict__['_data'].has_key(key):
            raise CatalogueError("Data table is read-only.")

        # otherwise set the class variable
        self.__dict__[key] = value

    def __getitem__(self, key):
        """ Return properties by name or return slices. """
        if type(key) is type("hi"):
            logging.warn("This behavior will be phased out. cat['key'] should be changed to cat.key.")
            return self._data[key]

        cat = self.__new__(self.__class__)

        # set attributes
        cat._data = self._data[key]
        cat._lon = self._lon[key]
        cat._lat = self._lat[key]
        cat._lon_name = self._lon_name
        cat._lat_name = self._lat_name
        cat._index = key
        return cat

    def __setitem__(self, key, cat):
        """ Set slices of the catalogue. """
        raise CatalogueError("Data table is read-only.")
        # if type(key) is type ("hi"):
        #     logging.warn("This behavior will be phased out. cat['key'] should be changed to cat.key.")
        #     self._data[key] = cat
        #     return
        # self._data[key] = cat._data
        # self._lon[key] = cat._lon
        # self._lat[key] = cat._lat
        # self._lookup_tree = None

    def __str__(self):
        """ helpful string representation """
        description = "Catalogue of " + str(len(self._data)) + " objects."
        return description

    def __len__(self):
        """ Return length of catalogue """
        return len(self._data)

    def __contains__(self, key):
        """ Check if we have the given field key """
        if self._data.dtype.fields is None: return False
        return self._data.dtype.fields.has_key(key)

    def make_view(self, cat):
        """ Copy data in from a catalogue. """
        self._data = cat._data
        self._lookup_tree = cat._lookup_tree
        self._lon_name = cat._lon_name
        self._lat_name = cat._lat_name
        self._lon = cat._lon
        self._lat = cat._lat

    def read(self, filename):
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
        pass

    def write(self, filename):
        """ write a data file """
        pass

    def build_tree(self):
        """ Initialize the data structure for fast spatial lookups.

        Inputs
        ------
        None

        Outputs
        -------
        None
        """
        xy = np.transpose([self._imagex, self._imagey])
        self._lookup_tree = KDTree(xy)

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
        try:
            self.lookup_tree
        except AttributeError:
            self.build_tree()

        if self.lookup_tree is None: self.build_tree()

        matches = self.lookup_tree.query_radius(np.transpose([x,y]), radius)
        return matches

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


    def plot(self):
        """ Create a Mollweide projected plot of the objects.

        Inputs
        ------
        None

        Outputs
        ------
        None
        """

        # Subsample a big catalogue
        if len(self) > 10000:
            index = np.arange(len(self))
            index = np.random.choice(index,10000)
            # Coordinates in radians
            x = misc.torad(self.lon[index])
            y = np.pi-(misc.torad(self.lat[index])+np.pi/2.0)

        # Setup healpix
        hp.graticule()
        hp.visufunc.projscatter(y, x, s=10., lw=0.0)


class CatalogueError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg



