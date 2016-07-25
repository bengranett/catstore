import numpy as np

import catalogue
import pypelid.vm.healpix_projection as HP
import pypelid.utils.filetools as filetools
import pypelid.utils.misc as misc
import pypelid.utils.sphere as sphere
from sklearn.neighbors import KDTree

import logging

class CatalogueStore(object):
    """ Storage backend for loading catalogues from disk.

    The catalogues support spatial queries using spherical coordinates. 
    
    Full-sky catalogues may be loaded in spatial chunks called zones only when needed so that the entire
    catalogue is not stored in memory.
    """

    # Define sky partition mode constants
    FULLSKY = 0
    HEALPIX = 1

    def __init__(self, filename=None, zone_resolution=2, zone_order=HP.RING,
                    check_hash=True, require_hash=True, official_stamp='pypelid'):
        """ Initialize the catalogue backend. """
        self.filename = filename
        self._trees = {}

        if filename is not None:
            self.init_pypelid_file(check_hash=check_hash, require_hash=require_hash, official_stamp=official_stamp)
            return
            
        self.zone_resolution = zone_resolution
        self.zone_order = zone_order
        self._hp_projector = HP.HealpixProjector(resolution = self.zone_resolution, order=self.zone_order)

    def init_pypelid_file(self, check_hash=True, require_hash=True, official_stamp='pypelid'):
        """ Check the input pypelid file and initialize. """
        filetools.validate_hdf5_file(self.filename, check_hash=check_hash, require_hash=require_hash, 
                                    official_stamp=official_stamp)

        with filetools.hdf5_catalogue(self.filename, mode='r') as h5file:
            # import all attributes in the file.
            self.metadata = {}
            for key,value in h5file.get_attributes().items():
                self.metadata[key] = value

            self.zone_resolution = self.metadata['zone_resolution']
            self.zone_order = self.metadata['zone_order']
        self._hp_projector = HP.HealpixProjector(resolution = self.zone_resolution, order=self.zone_order)
        self._datastore = None

    def __enter__(self):
        """ """
        self._open_pypelid(self.filename)
        return

    def __exit__(self, type, value, traceback):
        """ """
        self._close_pypelid()

    def _open_pypelid(self, filename):
        """ Load a pypelid catalogue store file. """
        self.h5file = filetools.hdf5_catalogue(filename, mode='r')
        # access the data group
        self._datastore = self.h5file.get_data()
        return self.h5file

    def _close_pypelid(self):
        """ """
        self.h5file.close()

    def write(self, filename):
        """ Construct and write a new catalogue store file in pypelid format. """
        pass

    def index(self):
        """ Generate the indices for the catalogue eg healpix zones. """
        pass

    def _retrieve_zone(self, zone):
        """ Retrieve the data within the given zones."""
        key = str(zone)
        return self._datastore[key]

    def _which_zones(self, lon, lat, radius):
        """ Determine which zones overlap the points (lon,lat) within the radius."""
        zones = self._hp_projector.query_disc(lon, lat, radius)
        logging.debug("querying %f,%f... zones found: %s",lon,lat,zones)
        return zones

    def _plant_tree(self, zone):
        """ Generate a KD tree data structure for the given zone to allow fast
        spatial queries.

        """
        data = self._retrieve_zone(zone)
        # access longitude and latitude...
        lon,lat = np.transpose(data['skycoord'][:])

        logging.debug("building tree %s n=%i",zone,len(lon))

        xyz = sphere.lonlat2xyz(lon, lat)
        self._trees[zone] = KDTree(np.transpose(xyz))

    def _query_cap(self, clon, clat, radius=1.):
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
        zones = self._which_zones(clon, clat, radius)

        matches = {}

        for zone_i in zones:

            if not self._trees.has_key(zone_i):
                self._plant_tree(zone_i)

            r = radius * np.pi/180

            xyz = sphere.lonlat2xyz(clon, clat)

            if misc.is_number(xyz[0]):
                xyz = np.transpose(xyz).reshape(1,-1)
            else:
                xyz = np.transpose(xyz)

            matches[zone_i] = self._trees[zone_i].query_radius(xyz, r)[0]

        return matches

    def _query_box(self,  clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
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
        match_dict = self._query_cap(clon,clat,r)

        selection_dict = {}
        for zone,matches in match_dict.items():
            data = self._retrieve_zone(zone)
            lon,lat = np.transpose(data['skycoord'][:][matches])
            dlon,dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)], inverse=True)

            sel_lon = np.abs(dlon) < (width/2. + pad_ra)
            sel_lat = np.abs(dlat) < (height/2. + pad_dec)
            sel = np.where(sel_lon & sel_lat)

            selection_dict[zone] = np.take(matches, sel[0])

        return selection_dict


    def retrieve(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0, transform=None):
        """ Select objects in a rectangle and return a projected catalogue object.

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
        structured array
        """
        data_dict = {}

        matches = self._query_box(clon, clat, width, height, pad_ra, pad_dec, orientation)

        for zone, selection in matches.items():
            data = self._retrieve_zone(zone)
            for name,arr in data.items():
                if not name in data_dict:
                    data_dict[name] = []
                data_dict[name].append(arr[:][selection])

        for name in data_dict.keys():
            data_dict[name] = np.concatenate(data_dict[name])

        # construct a structured array
        structured_arr = misc.dict_to_structured_array(data_dict)

        return structured_arr


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

