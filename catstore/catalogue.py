import logging
import numpy as np
import fitsio
from sklearn.neighbors import KDTree
from pypelid.utils import sphere, misc
from vm.healpix_projection import HealpixProjector

class Catalogue(object):
    """ Base catalogue class. """

    _lon_name = 'alpha'
    _lat_name = 'delta'

    def __init__(self, cat=None, data=None, zone_resolution=5, hp_order='nest'):
        """
        Inputs
        ------
        cat - take data from this catalog if given.
        """
        if cat is not None:
            self.make_view(cat)
            return
        elif data is not None:
            self._data = data
        else:
            self._data = np.array([])

        self._hp_projector = HealpixProjector(2**zone_resolution, hp_order)

        # copy lon and lat arrays if we have them
        if self.__contains__(self._lon_name):
            self._lon = self._data[self._lon_name].copy()
        if self.__contains__(self._lat_name):
            self._lat = self._data[self._lat_name].copy()

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
                self._data = np.genfromtxt(filename, names=names, converters=converters)
            except ValueError as e:
                print e.message
                raise Exception("The input catalogue %s is in the wrong format. Run prepcat first!"%filename)

        elif filename.endswith('.fits'):
            try:
                self._data = fitsio.read(filename, columns=names, ext=fits_ext)
            except ValueError as e:
                print e.message
                raise Exception("The input catalogue %s is in the wrong format. Run prepcat first!"%filename)

        else:
            raise Exception("Unknown file type: %s"%filename)

        assert(self._data.dtype.fields.has_key(self._lon_name))
        assert(self._data.dtype.fields.has_key(self._lat_name))

        self._lon = self._data[self._lon_name].copy()  # copy because these coordinates can be transformed
        self._lat = self._data[self._lat_name].copy()

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
        xyz = sphere.lonlat2xyz(self._lon, self._lat)
        self._lookup_tree = KDTree(np.transpose(xyz))


    def projection(self, transform):
        """ Apply spatial projection. """
        self._transform = transform
        return self
        #self._lon, self._lat = transform(self._lon, self._lat)

    #def to_cartesian(self):
    #    """ Return a cartesian catalogue """
    #    return CartesianCatalogue(self)

    def _get_zones_of_interest(self, lon, lat, radius):
        """ Determine the zones that overlap the points (lon,lat) within the radius."""
        raise Exception("Not implemented")
        # pix = np.arange(self._healpix_projector.npix)
        # grid = self._healpix_projector.pix2ang(pix)
        # lookup_tree = KDTree(grid)

        # self._healpix_projector.ang2pix(lon, lat)

        return zones

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
        zones = self._get_zones_of_interest(clon, clat, radius)

        matches = []

        for zone_i in zones:

            if not self._trees.has_key(zone_i):
                self._build_tree(zone_i)

            r = radius * np.pi/180

            xyz = sphere.lonlat2xyz(clon, clat)

            if misc.is_number(xyz[0]):
                xyz = np.transpose(xyz).reshape(1,-1)
            else:
                xyz = np.transpose(xyz)

            matches.append(self._trees[zone_i].query_radius(xyz, r))

        return np.concatenate(matches)

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

        lon = self._lon[cap]
        lat = self._lat[cap]

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
        return self.__getitem__(matches)

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
        try:
            self.lookup_tree
        except AttributeError:
            self.build_tree()

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


class CatalogueError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg



