import numpy as N
import h5py

import catalogue
import pypelid.vm.healpix_projection as HP


class CatalogueStore(object):
    """ Storage backend for loading catalogues from disk.

    The catalogues support spatial queries using spherical coordinates. 
    
    Full-sky catalogues may be loaded in spatial chunks called zones only when needed so that the entire
    catalogue is not stored in memory.
    """

    # Define sky partition mode constants
    FULLSKY = 0
    HEALPIX = 1

    def __init__(self, filename=None, zone_resolution=2, zone_order=HP.RING):
        """ Initialize the catalogue backend. """
        self.zone_resolution = zone_resolution
        self.zone_order = zone_order
        self.filename = filename

        if filename is not None:
            self._datastore = self.load(filename)

        self._hp_projector = HP.HealpixProjector(2**self.zone_resolution, self.hp_order)


    def load(self, filename):
        """ Load a catalogue from disk.  Multiple formats are supported to allow easy
        conversion for preprocessing catalogues.

        The file format is guessed from the extension.
        """
        if filename.endswith("fits"):
            self.load_fits(filename)
        elif filename.endswith("dat"):
            self.load_ascii(filename)
        elif filename.endswith("hdf5") or filename.endswith("pypelid"):
            self.load_hdf5(filename)
        else:
            raise Exception("Unknown file format: %s"%filename)

    def load_hdf5(self, filename):
        """ Load a pypelid catalogue store file. """
        datastruct = h5py.File(filename)
        self.partition_mode = datastruct['partition_mode']
        self.zone_resolution = datastruct['zone_resolution']
        self.zone_order = datastruct['zone_order']
        self._datastore = datastruct
        self.filename = datastruct.filename
        self.name = datastruct.name

    def load_fits(self, filename, names=None, fits_ext=1):
        """ Load a fits data file. """
        try:
            data, fits_header = fitsio.read(filename, columns=names, ext=fits_ext, header=True)
            name = self.header['EXTNAME']
        except ValueError as e:
            print e.message
            raise Exception("The input catalogue %s is in the wrong format. Run prepcat first!"%filename)

        self._datastore = data
        self.name = name
        self.partition_mode = self.FULLSKY
        self.zone_resolution = 0
        self.zone_order = None

    def load_ascii(self, filename):
        """ Load data in ascii format """
        raise Exception("Not implemented")

    def write(self, filename):
        """ Construct and write a new catalogue store file in pypelid format. """
        pass

    def index(self):
        """ Generate the indices for the catalogue eg healpix zones. """
        pass

    def _retrieve_zone(self, zones):
        """ Retrieve the data within the given zones."""
        data = []
        for zone in zones:
            key = str(zone)
            data.append(self._datastore[key])
        data = N.vstack(data)
        return data

    def _which_zones(self, lon, lat, radius):
        """ Determine which zones overlap the points (lon,lat) within the radius."""

        zones = self._healpix_projector.query_disc(lon, lat, radius)
        return zones

    def _plant_tree(self, zone):
        """ Generate a KD tree data structure for the given zone to allow fast
        spatial queries.

        """
        data = self._retrieve_zone(zone)
        # access longitude and latitude...
        lon = data['lon']
        lat = data['lat']
        xyz = sphere.lonlat2xyz(lon, lat)
        self._tree[zone] = KDTree(np.transpose(xyz))

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

            matches[zone_i] = self._trees[zone_i].query_radius(xyz, r)

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

        lon = self._lon[cap]
        lat = self._lat[cap]

        dlon,dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)], inverse=True)

        sel_lon = np.abs(dlon) < (width/2. + pad_ra)
        sel_lat = np.abs(dlat) < (height/2. + pad_dec)
        sel = np.where(sel_lon & sel_lat)

        matches = np.take(cap, sel[0])
        return matches


    def project_subcat(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0, transform=None):
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
        catalogue
        """
        matches = self.query_box(clon, clat, width, height, pad_ra, pad_dec, orientation)
        # return a new catalogue instance that contains the selected objects...
        # return catalogue


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

