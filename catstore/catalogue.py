# -*- coding: utf-8 -*-
#!/usr/bin/env python

import logging
import numpy as np
from sklearn.neighbors import KDTree
from catstore import hdf5tools, _querycat, misc, sphere

class Catalogue(object):
    """ Base catalogue class. Internal to Pypelid. """
    logger = logging.getLogger(__name__)

    _coordinate_system = 'equatorial'
    _required_columns = ()
    _required_meta = ()
    _immutable_attributes = ('_coordinate_system', '_required_columns', '_required_meta', '_immutable_attributes')

    def __init__(self, data=None, scale_x=1, scale_y=1, angle=0, metadata=None, **attrs):
        """
        Parameters
        ----------
        data : recarray
            structured array, rows are objects, columns are object properties, which include at least:
            x, y, mag, ra, dec
        metadata : dict
            dictionary of metadata, should include at least:
            catalogue name, zone_resolution, zone_list, a centre tuple
        """
        # process the input meta data
        meta = {}
        if metadata is not None:
            for key, value in list(metadata.items()):
                meta[key] = value
        for key, value in list(attrs.items()):
            if key not in ['imagecoord','skycoord']:
                meta[key] = value

        for key in self._required_meta:
            if key not in meta:
                raise Exception('Meta key %s not found. It is needed in Catalogue metadata!'%key)

        #if len(meta) > len(self._required_meta):
        #   logging.warning('Some of the metadata you are passing to Catalogue is not used.')

        self.__dict__['_meta'] = meta

        # Save column names for practicality
        self.columns = None

        # Load the data
        self.__dict__['_data'] = {}
        if data is not None:
            self.load(data)

        self.__dict__['_spatial_key'] = None
        try:
            self._data['imagecoord']
            self._spatial_key = 'imagecoord'
        except (KeyError, ValueError):
            # First try to find coordinates in the data structure.
            # If the coordinates are not in the input data structure,
            # check if they have been input as a separate array.
            try:
                self._data['skycoord']
                self._spatial_key = 'skycoord'
            except KeyError:
                # Look at the attrs, if they contain any kind of coordinates.
                if 'imagecoord' in list(attrs.keys()):
                    self._spatial_key = 'imagecoord'
                elif 'skycoord' in list(attrs.keys()):
                    self._spatial_key = 'skycoord'
                else:
                    # This only logs the error - it does not raise an Exception.
                    self.logger.error("Need imagecoord or skycoord to make spatial queries.")
                # In case no data array has been input, only the coordinates as a column
                if data is None and len(attrs)>0:
                    self.load({self._spatial_key: attrs[self._spatial_key]})
        # self.logger.debug("Using %s for spatial index", self._spatial_key)
        self.__dict__['_scale'] = (scale_x, scale_y, angle)
        self.__dict__['Query'] = None

    def __getstate__(self):
        """ """
        state = (
            self.__dict__['_meta'],
            self.__dict__['_data'],
            self.__dict__['_spatial_key'],
            self.__dict__['_scale'],
            self.__dict__['columns'],
        )
        return state

    def __setstate__(self, state):
        """ """
        meta, data, spatial_key, scale, columns = state
        self.__dict__['_meta'] = meta
        self.__dict__['_data'] = data
        self.__dict__['_spatial_key'] = spatial_key
        self.__dict__['_scale'] = scale
        self.__dict__['columns'] = columns
        scale_x, scale_y, angle = scale
        self.Query = None

    def __getattr__(self, key):
        """ Return columns by name

        Columns may be accessed by catalogue.alpha

        Parameters
        ----------
        key

        """
        # if the attribute is a class variable, return it
        try:
            return self.__dict__[key]
        except KeyError:
            pass

        # if the attribute is a column name in the data table, return the column
        try:
            return self.__dict__['_data'][key]
        except KeyError as e:
            raise AttributeError(e.message)
        except IndexError as e:
            raise AttributeError(e.message)
        except ValueError as e:
            raise AttributeError(e.message)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setattr__(self, key, value):
        """ Set a catalogue attribute.

            Raises CatalogueError exception if the variable name exists in the data table.

        Parameters
        ----------
        key
        value

        """
        try:
            self.__dict__['_data'][key]
            raise CatalogueError("Data table is read-only.")
        except KeyError:
            pass
        except ValueError:
            pass
        if key in self._immutable_attributes:
            raise CatalogueError(str(key) + ' is read-only.')

        # otherwise set the class variable
        self.__dict__[key] = value

    def __str__(self):
        """ helpful string representation """
        description = "Catalogue of " + str(len(self._data)) + " objects."
        return description

    def __len__(self):
        """ Return length of catalogue """
        try:
            return self._len
        except AttributeError:
            self._len = len(self._data)
        return self._len

    def __contains__(self, key):
        """ Check if we have the given field key """
        if self._data.dtype.fields is None: return False
        return key in self._data.dtype.fields

    def _convert_CatStore(self, store):
        """ """
        arr = store.to_structured_array()
        self.columns = list(arr.dtype.names)
        self.__dict__['_data'] = arr

    def init_query(self):
        """ """
        scale_x, scale_y, angle = self.__dict__['_scale']
        self.Query = _querycat.QueryCat(self.__dict__['_data'][self._spatial_key][:], scale_x=scale_x, scale_y=scale_y, angle=angle)


    def load(self, data):
        """ Import the data array into the Catalogue class.

            Parameters
            ----------
            data : dict or numpy.recarray
                the input data table
        """
        # Test if conversion to structured array is needed

        convert = False

        try:
            self._convert_CatStore(data)
            return
        except:
            pass

        try:
            # Check if the input data is a structured array
            if not data.dtype.fields:
                raise TypeError('The Catalogue class must be loaded with a dictionary-like structure or numpy structured array!')
            else:
                # Save column names for practicality
                self.columns = list(data.dtype.names)
        except AttributeError:

            # Check if the input data is a dict
            try:
                list(data.items())
            except AttributeError:
                raise TypeError('The Catalogue class must be loaded with a dictionary-like structure or numpy structured array!')
            else:
                convert = True
                # Save column names for practicality
                self.columns = list(data.keys())

        # Make sure the needed columns are loaded
        for column in self._required_columns:
            try:
                data[column]
            except KeyError:
                raise Exception('Data array is missing a required column: %s'%column)

        if isinstance(data, hdf5tools.Group):
            self.__dict__['_data'] = data
            return

        if convert:
            self.__dict__['_data'] = misc.dict_to_structured_array(data)
        else:
            self.__dict__['_data'] = data

    def dump(self, filename):
        """ Dump the data array to a Hickle file. """
        import hickle
        hickle.dump(filename, self.__dict__['_data'])

    def get_data(self):
        """ Return the data array."""
        return self.__dict__['_data']

    def get_meta(self, key):
        """ """
        return self.__dict__['_meta'][key]



    def query_disk(self, x, y, radius=1.):
        """ Find neighbors to a given point (ra, dec).

        Parameters
        ----------
        x
            center x
        y
            center y
        radius : optional
            radius (default=1.)

        Returns
        -------
        ?
            indices of objects in selection

        """
        if not self.Query:
            self.init_query()

        return self.Query.query_disk(np.transpose([x,y]), radius)

    def query_box(self,  cx, cy, width=1, height=1, pad_x=0.0, pad_y=0.0, orientation=0.):
        """ Find objects in a rectangle.

        Parameters
        ----------
        x : array
            center x
        y : array
            center y
        width : float or numpy.array
            width
        height : float or numpy.array
            height
        pad_x : float
            add this padding to width
        pad_y : float
            add this padding to height
        orientation : float
            position angle of the box

        Returns
        -------
        results : list of lists
            list of indices of objects in selection

        """
        if not self.Query:
            self.init_query()

        n = len(cx)
        ones = np.ones(n, dtype=float)

        matches, eff = self.Query.query_box(
                        np.transpose([cx, cy]),
                        width=width*ones,
                        height=height*ones,
                        pad_x=pad_x,
                        pad_y=pad_y,
                        orientation=orientation
                )
        self.logger.debug("Query efficiency %f", eff)
        return matches

    def retrieve(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0,
                    orientation=0, transform=None, columns=None):
        """ Select objects in a rectangle and return a projected catalogue object.

        Parameters
        ----------
        clon : float
            center ra
        clat : float
            center dec
        width : float
            rectangle width (degrees)
        height : float
            rectangle height (degrees)
        pad_ra : float
            add this padding to width (degrees)
        pad_dec : float
            add this padding to dec (degrees)
        orientation : float
            rotation angle (degrees about center from North increasing toward the East)
        transform : function
            a function that will be called to map sky coordinates to cartesian coordinates.

        Returns
        -------
        Catalogue : Catalogue object
        """
        lon,lat = np.transpose(self._data['skycoord'])
        dlon, dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)],
                                    inverse=True)

        sel_lon = np.abs(dlon) < (width / 2. + pad_ra)
        sel_lat = np.abs(dlat) < (height / 2. + pad_dec)
        sel = sel_lon & sel_lat
        data = self._data[sel]
        if transform is not None:
            ximage, yimage = transform(lon[sel], lat[sel])
            data['imagecoord'] = np.transpose([ximage, yimage])
        return Catalogue(data=data, selection=sel)

    def update(self, arr, sel=True, operation='replace', columns=None):
        """ """
        if not columns:
            columns = [name for name in arr.dtype.names if not name.startswith("_")]

        logging.debug("update with %s columns %s", operation, str(columns))

        for name in columns:
            if operation == 'sum':
                self.__dict__['_data'][name][sel] += arr[name]
            elif operation == 'replace':
                self.__dict__['_data'][name][sel] = arr[name]
            else:
                raise ValueError("update operation must be 'replace' or 'sum' (unknown %s)", operation)

    def plot(self, nplot=10000, **plotparams):
        """ Make a cartesian image coordinates scatter plot using matplotlib.

        Parameters
        ----------
        nplot : int
            Maximum number of objects to show in the plot.

        """
        xy = self.__dict__['_data'][self._spatial_key][:]

        # Subsample a big catalogue
        if len(self) > nplot:
            index = np.arange(len(self))
            index = np.random.choice(index, nplot)
            # Coordinates in radians
            x,y = np.transpose(xy[index])
        else:
            x,y = np.transpose(xy)

        import pylab
        pylab.scatter(x,y,**plotparams)

class CatalogueError(Exception):
    pass
