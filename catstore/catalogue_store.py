import numpy as N
import h5py

import catalogue


class CatalogueStore(object):
    """ Storage backend for catalogues. """

    def __init__(self, filename=None):
        """ """
        if filename is not None:
            self.store = self.load(filename)

    def load_hdf5(filename):
        """ Load a catalogue store file. """
        self.store = h5py.File(filename)

    def load_fits(filename):
        """ Load a fits data file. """
        raise Exception("Not implemented")

    def load_ascii(filename):
        """ Load data in ascii format """
        raise Exception("Not implemented")

    def save(filename):
        """ Construct and write a new catalogue store file. """
        self.store = h5py.File(filename)


    def _retrieve_data(self, zones):
        """ Retrieve the data within the given zones."""
        raise Exception("Not implemented")
        # for zi in zones:
        #     self._data_storage[str(zi)]

    def _which_zones(self, lon, lat, radius):
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
        """ Select objects in a rectangle and return a catalogue object.

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
