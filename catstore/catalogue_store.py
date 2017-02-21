import os
import numpy as np
import copy
import catalogue
import pypelid.utils.healpix_projection as HP
import pypelid.utils.hdf5tools as hdf5tools
import pypelid.utils.misc as misc
import pypelid.utils.sphere as sphere

import logging


class CatalogueStore(object):
	""" Catalogue storage backend using HDF5.

	The catalogue data should be accessed with spatial queries through the retrieve() method.

	Under the hood the survey is partitioned into zones using Healpix cells.  The resolution
	is set by the zone_resolution parameter.

	Additional name=value arguments will be stored in the meta data.

	Parameters
	----------
	filename : str
		Path to file.  If exists file will be opened read-only.
	mode : str
		Mode for opening file ('r', 'w').  Existing files are always opened read-only.
	zone_resolution : int
		Zone resolution to use for partitioning data groups. resolution = log2(healpix nside)

	Other Parameters
	----------------
	zone_order : str
		Zone ordering can be either healpix ring or nest ordering. (either 'ring' or 'nest')
		Ring ordering is faster for querying.  (Default ring)
	check_hash : bool
		If True, check hash when opening file. (Default True)
	require_hash : bool
		Raise error if hash does not validate. (Default True)
	official_stamp : str
		String used to prefice catalogue files. (Default 'pypelid')

	Notes
	--------
	API to use to create a new catalogue file.

		>>> data = {
		>>>         'skycoord': coord,
		>>>         'redshift': redshift,
		>>>        }
		>>> dtypes = [
		>>>           np.dtype([('skycoord', float, 2)]),
		>>>           np.dtype([('redshift', float, 1)])
		>>>          ]
		>>> units = {'skycoord': 'degree', 'redshift': 'redshift'}
		>>> meta = {'coordsys': 'equatorial J2000'}
		>>> with CatalogueStore(filename, 'w', name='test') as cat:
		>>>     cat.preload(ra, dec)
		>>>     cat.allocate(dtypes)
		>>>     cat.load(data)
		>>>     cat.load_attributes(meta)
		>>>     cat.load_units(units)
		>>>     cat.load_description(description)

	"""

	logger = logging.getLogger(__name__)

	# Define sky partition mode constants
	ZONE_ZERO = 0
	FULLSKY = 'FULLSKY'
	HEALPIX = 'HEALPIX'
	_required_attributes = {'partition_scheme': (HEALPIX,),
							'zone_resolution': range(12),
							'zone_order': (HP.RING, HP.NEST)
							}
	_required_columns = ('skycoord',)

	_default_params = {'zone_resolution': 1,
						'partition_scheme': HEALPIX,
						'zone_order': HP.RING,
						'check_hash': True,
						'require_hash': True,
						'official_stamp': 'pypelid',
						'preallocate_file': True,
						'overwrite': False,
						}

	def __init__(self, filename, mode='r', preallocate_file=True, overwrite=False, **input_params):
		""" """
		self.params = copy.deepcopy(self._default_params)

		self.params['preallocate_file'] = preallocate_file
		self.params['overwrite'] = overwrite

		for key, value in input_params.items():
			try:
				self.params[key] = value
			except KeyError:
				self.logger.warning("Unrecognized argument passed to CatalogueStore (%s)" % key)

		self._check_inputs()

		self._h5file = None
		self._datastore = None
		self._attributes = {}

		self.filename = filename

		self.readonly = False

		if mode not in ('r', 'w', 'a'):
			raise CatStoreError("Invalid argument: mode must be one of ('r', 'w', 'a')")

		if os.path.exists(filename):
			if mode == 'w':
				if not self.params['overwrite']:
					raise CatStoreError("File %s exists.  Will not overwrite."%filename)
			else:  # mode is r or a
				if mode == 'r':
					self.readonly = True
				self._load_pypelid_file()
				self._open_pypelid(filename, mode=mode)
				return

		self._hp_projector = HP.HealpixProjector(resolution=self.params['zone_resolution'],
												order=self.params['zone_order'])

		self._open_pypelid(filename, mode=mode)
		self.load_attributes(partition_scheme=self.params['partition_scheme'],
							zone_resolution=self.params['zone_resolution'],
							zone_order=self.params['zone_order'])

	def _check_inputs(self):
		""" Check inputs are valid """
		assert self.params['partition_scheme'] in (self.HEALPIX, self.FULLSKY)
		assert self.params['check_hash'] in (True, False)
		assert self.params['require_hash'] in (True, False)
		assert self.params['preallocate_file'] in (True, False)
		assert self.params['overwrite'] in (True, False)
		assert isinstance(self.params['official_stamp'], basestring)
		HP.validate_resolution(self.params['zone_resolution'])
		HP.validate_order(self.params['zone_order'])

	def _load_pypelid_file(self):
		""" Check the input pypelid file and initialize.
		"""
		hdf5tools.validate_hdf5_file(self.filename, check_hash=self.params['check_hash'],
									require_hash=self.params['require_hash'],
									official_stamp=self.params['official_stamp'])

		with hdf5tools.HDF5Catalogue(self.filename, mode='r', overwrite=self.params['overwrite']) as h5file:

			# ensure that required attributes are there with acceptable values.
			for key, options in self._required_attributes.items():
				try:
					assert(h5file.attributes[key] in options)
				except KeyError:
					raise Exception("Cannot load %s: attribute is missing: %s"%(self.filename,key))
				except AssertionError:
					raise Exception("Cannot load %s: invalid attribute: %s:%s (expected %s)"%(self.filename,key,h5file.attributes[key],options))

			# ensure that required datasets are there
			for column_name in self._required_columns:
				try:
					if column_name not in h5file.get_columns():
						raise Exception("Cannot load %s: data column is missing: %s" % (self.filename, column_name))
				except KeyError:
					# The file may not be loaded yet, so ignore this error.
					pass

			self._hp_projector = HP.HealpixProjector(resolution=h5file.attributes['zone_resolution'],
													order=h5file.attributes['zone_order'])
		self._datastore = None

	def __str__(self):
		return self.filename

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.close()

	def __getitem__(self, key):
		""" Get a zone by name. """
		return self._datastore[key]

	def __getattr__(self, key):
		""" """
		if key == 'columns':
			return self.get_columns()
		elif key == 'dtype':
			return self.get_dtype()

		try:
			return self._attributes[key]
		except KeyError:
			return self._metadata[key]

	def __iter__(self):
		""" """
		self.cat_metadata = {}
		for key, value in self._attributes.items():
			self.cat_metadata[key] = value
		self._iter_zone_index = 0
		self._iter_zone_list = self._datastore.keys()
		return self

	def __len__(self):
		return len(self._datastore)

	def next(self):
		""" """
		try:
			zone = self._iter_zone_list[self._iter_zone_index]
			self._iter_zone_index += 1

			# Include zone index into the metadata so it can be looked up later
			metadata = dict(self.cat_metadata.items()+[('zone', int(zone))])
			return catalogue.Catalogue(data=self._datastore[zone], metadata=metadata)

		except IndexError:
			raise StopIteration()

	def _open_pypelid(self, filename, mode='r'):
		""" Load a pypelid catalogue store file. """
		self._h5file = hdf5tools.HDF5Catalogue(filename,
											mode=mode,
											preallocate_file=self.params['preallocate_file'],
											overwrite=self.params['overwrite'])
		# access the data group
		self._datastore = self._h5file.data
		self._metadata = self._h5file.metadata

		try:
			self._attributes = self._h5file.attributes
		except:
			self._attributes = {}
			raise

		if mode is not 'r':
			if 'zone_counts' not in self._metadata:
				self._metadata['zone_counts'] = np.zeros(self._hp_projector.npix, dtype=int)

			if 'allocation_done' not in self._metadata:
				self._metadata['allocation_done'] = False

			if 'preallocate_file' not in self._metadata:
				self._metadata['preallocate_file'] = self.params['preallocate_file']

			if 'done' not in self._metadata:
				self._metadata['done'] = False

		return self._h5file

	def close(self):
		""" """
		if self._h5file is not None:
			self._h5file.close()

	def preprocess(self, lon, lat, dtypes=None):
		""" Run pre-processing needed before creating a new catalogue file.

		Given longitude and latitude coordinates count number of objects in each zone
		for pre-allocation.

		Parameters
		----------
		lon : ndarray
			longitude
		lat : ndarray
			latitude

		Returns
		-------
		None
		"""
		zone_counts = self._metadata['zone_counts']

		zid = self._index(lon, lat)
		counts = np.bincount(zid)
		zone_counts[:len(counts)] += counts

		self._metadata['zone_counts'] = zone_counts

	# For backward compatability rename preprocess to preload
	preload = preprocess

	def allocate(self, dtypes):
		""" Pre-allocate the file.

		Parameters
		----------
		dtypes : list of numpy dtypes
		"""
		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return

		zone_counts = self._metadata['zone_counts']
		index, = np.where(zone_counts)
		self.logger.debug("Preallocating group IDs: %s", index)
		self._h5file.preallocate_groups(index, zone_counts[index], dtypes=dtypes)
		self._metadata['allocation_done'] = True

	def load(self, data):
		""" Add data to the file.

		Parameters
		----------
		data : dict or numpy structured array
		"""
		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return

		try:
			data['skycoord']
		except KeyError, ValueError:
			raise Exception("skycoord column is required")

		zone_index = self._index(*data['skycoord'].transpose())
		self._h5file.update(zone_index, data)

		self._datastore = self._h5file.data

		try:
			self._attributes['count']
		except KeyError:
			self._attributes['count'] = 0


		# Save the number of rows in table (i.e. total number of objects in the catalogue)
		if isinstance(data, dict):
			key, arr = data.items()[0]
			count = len(arr)
		else:
			count = len(data)
		self._attributes['count'] += count

	def load_attributes(self, attrib=None, **args):
		""" Update file metadata.

		attrib: dict
			key-value pairs
		"""
		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return
		self._h5file.update_attributes(attrib, **args)

	def load_units(self, attrib):
		""" Update units in file metadata.

		attrib : dict
			dictionary with column names and units
		"""
		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return
		self._h5file.update_units(attrib)

	def load_description(self, attrib):
		""" Update description in file metadata.

		attrib : dict
			dictionary with column names and description
		"""
		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return
		self._h5file.update_description(attrib)

	def update(self, data):
		""" Updates the data that has already been loaded.
			This should not be overwriting the input catalogue columns.
			This is only for saving results!
			
			TODO: raise Exception if input columns update is attempted.

			Parameters
			----------
			data : numpy.array
				This structured array contains the values to save and the which 
				row to find the object in: 'index' and 'skycoords'.

			Returns
			-------
			None
		"""

		if isinstance(data,dict):
			data = misc.dict_to_structured_array(data)
		logging.debug('Updating CatalogueStore object with data of type: ' + str(data.dtype))

		if self.readonly:
			self.logger.warning("File is readonly! %s", self.filename)
			return

		if ('skycoord' not in data.dtype.names and 'zone' not in data.dtype.names) or 'index' not in data.dtype.names:
			raise Exception("skycoord or zone and index columns are required")
		
		# Get the zone index
		if 'zone' not in data.dtype.names:
			zone_index = self._index(*data['skycoord'].transpose())
		else:
			zone_index = data['zone']

		# Which columns to update
		columns = [col for col in data.dtype.names if col not in ['skycoord', 'zone']]

		# This updates all the rows in the column, since that is what we need here to save results
		self._h5file.update(zone_index, data[columns], index=data['index'], ind_col='index')

	def _index(self, lon, lat):
		""" Generate the indices for the catalogue eg healpix zones. """
		if self._attributes['partition_scheme'] == self.HEALPIX:
			return self._hp_projector.ang2pix(lon, lat)
		return self.ZONE_ZERO

	def _retrieve_zone(self, zone):
		""" Retrieve the data within the given zones."""
		key = str(zone)
		if key not in self._datastore:
			raise ZoneDoesNotExist()
		return self._datastore[key]

	def _which_zones(self, lon, lat, radius):
		""" Determine which zones overlap the points (lon,lat) within the radius."""
		zones = self._hp_projector.query_disc(lon, lat, radius)
		self.logger.debug("querying %f,%f... zones found: %s", lon, lat, zones)
		return zones

	def get_zones(self):
		""" Return a list of zone identifiers. """
		return self._datastore.keys()

	def get_data(self):
		""" A generating function that returns the hdf5 groups."""
		self.logger.warning("get_data is deprecated.  Use iterator instead.")
		metadata = {}
		for key, value in self._attributes.items():
			metadata[key] = value

		for zone in self.get_zones():
			# construct a catalogue object
			cat = catalogue.Catalogue(data=self._datastore[zone],
								metadata=metadata)
			yield cat

	def get_attribute(self, key):
		""" Return the attribute by name.

		Parameters
		----------
		key : str
			Name of attribute to return.
		"""
		return self._attributes[key]

	def get_columns(self):
		""" Return a tuple of column names
		"""
		try:
			return tuple([name for name in self._h5file.get_columns()])
		except KeyError:
			return ()

	def get_dtype(self):
		""" Return the column/dataset descriptions as a structured-array-like dtype object.
		"""
		dtypes = []
		for zone in self._datastore.keys():
			# get a data group (doesn't matter which)
			for name, dataset in self._datastore[zone].items():
				# dim is the number of elements of each entry
				dim = dataset.shape[1:]
				# hdf5 group names are returned in unicode
				dtypes.append((name.encode('ascii'), dataset.dtype, dim))
			if len(dtypes) > 0:
				# if we got something, break out
				break

		return np.dtype(dtypes)

	def to_structured_array(self, columns=None, zones=None):
		""" Convert the data stored in the CatalogueStore to a numpy structured array.

		Parameters
		----------
		columns : list
			list of column names to access.  If None, all will be returned
		zones : list
			list of zone names in CatalogueStore to access.  If None, all will be returned.
			The list of zone names can be read with get_zones()

		Outputs
		---------
		numpy structured array
		"""
		dtype = self.get_dtype()
		if columns is not None:
			d = []
			for name in columns:
				d.append((name.encode('ascii'), dtype[name]))
			dtype = np.dtype(d)

		if zones is None:
			# retrieve all zones
			zones = self.get_zones()
			count = self.count
		else:
			# Need to count number of objects in the specified zones
			count = 0
			for zone in zones:
				count += self._retrieve_zone(zone).attrs['count']

		struc_array = np.zeros(count, dtype=dtype)

		i = 0
		for zone in zones:
			group = self._retrieve_zone(zone)
			count = group.attrs['count']
			j = i + count

			if columns is None:
				columns = group.keys()

			for name in columns:
				column = group[name]
				struc_array[name.encode('ascii')][i:j] = column

			i = j

		assert j == len(struc_array)

		return struc_array

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

		mu_thresh = np.cos(radius * np.pi / 180)

		xyz = sphere.lonlat2xyz(clon, clat)

		if misc.is_number(xyz[0]):
			xyz = np.transpose(xyz).reshape(1, -1)
		else:
			xyz = np.transpose(xyz)

		matches = {}
		count = 0
		for zone_i in self._which_zones(clon, clat, radius):
			try:
				data = self._retrieve_zone(zone_i)
			except ZoneDoesNotExist:
				continue

			# access longitude and latitude...
			lon, lat = np.transpose(data['skycoord'])
			cat_xyz = sphere.lonlat2xyz(lon, lat)

			mu = np.dot(xyz, cat_xyz)
			cut = mu > mu_thresh
			cut = np.sum(cut, axis=0)          # OR operation along the first dimension
			matches[zone_i] = np.where(cut)
			count += len(matches[zone_i][0])

		return matches, count

	def _query_box(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
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
		r = np.sqrt(width**2 + height**2) / 2.
		match_dict, count_cap = self._query_cap(clon, clat, r)

		selection_dict = {}
		count = 0
		for zone, matches in match_dict.items():
			data = self._retrieve_zone(zone)
			lon, lat = np.transpose(data['skycoord'][:][matches])
			dlon, dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)],
												inverse=True)

			sel_lon = np.abs(dlon) < (width / 2. + pad_ra)
			sel_lat = np.abs(dlat) < (height / 2. + pad_dec)
			sel = np.where(sel_lon & sel_lat)

			selection_dict[zone] = np.take(matches, sel[0])

			count += len(sel[0])

		return selection_dict, count


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

		# Raise error if data has not yet been loaded
		if self._datastore is None:
			raise Exception('You cannot retrieve from an empty catalogue!')

		matches, count = self._query_box(clon, clat, width, height, pad_ra, pad_dec, orientation)

		logging.debug('Found ' + str(count) + ' objects at (' + str(clon) + ',' + str(clat) + ').')

		# Return none if no objects fall into the rectangle
		if len(matches) == 0:
			return None

		# construct the dtype for requested columns
		dtype = self.get_dtype()
		if columns is not None:
			d = []
			for name in columns:
				d.append((name, dtype[name]))
			dtype = np.dtype(d)

		# add imagecoord and zone columns to dtype
		dtype = misc.concatenate_dtypes([dtype, 
										 np.dtype([('imagecoord', float, 2)]),
										 np.dtype([('zone', int, 1)])])

		# allocate a structured array
		struc_array = np.zeros(count, dtype=dtype)

		# zone start
		i = 0
		for zone, selection in matches.items():
			data = self._retrieve_zone(zone)

			if columns is None:
				columns = data.keys()

			# zone end
			j = i+count
			for column in columns:
				arr = data[column][:][selection]
				assert count==len(arr)
				struc_array[column][i:j] = arr

			# Add a column that contains the zone information
			struc_array['zone'][i:j] = zone

			# next zone start
			i += count

		if transform is not None:
			lon, lat = np.transpose(struc_array['skycoord'])
			ximage, yimage = transform(lon, lat)
			struc_array['imagecoord'] = np.transpose([ximage, yimage])
		else:
			struc_array['imagecoord'] = struc_array['skycoord']


		# construct a catalogue object
		metadata = {}
		for key, value in self._attributes.items():
			metadata[key] = value
		cat = catalogue.Catalogue(data=struc_array,
								metadata=metadata,
								center=(clon, clat))

		return cat

	def plot(self, plot_every=1, s=5., lw=0., cmap='jet'):
		""" Create a Mollweide projected plot of the objects.

		Uses healpy for the spherical projection and matplotlib for the colormap.

		Parameters
		----------
		plot_every : int
				Downsample the points before plotting.
		s : float
				Marker size
		lw : float
				Line width
		cmap : colormap
				A matplotlib colormap.
		"""
		from matplotlib.cm import ScalarMappable
		from matplotlib.colors import Normalize
		import healpy as hp

		sc = ScalarMappable(Normalize(0, len(self.get_zones())), cmap=cmap)

		# initialize a mollweide map
		tot = 0
		counter = 0
		hp.mollview(np.zeros(12) + float('nan'), cbar=False)
		for data in self:
			lon, lat = np.transpose(data['skycoord'])
			counter += 1
			print lon.shape
			# select a subset of points
			if plot_every > 1:
				sel = np.random.choice(len(lon), len(lon) // plot_every)
				lon = lon[sel]
				lat = lat[sel]

			# color points based upon zone index
			c = sc.to_rgba(int(data.name.split("/")[-1]))

			# plot
			hp.visufunc.projscatter(lon, lat, c=c, s=s, lw=lw, lonlat=True)
			tot += len(lon)
		hp.graticule()

		print "number of groups",counter
		print "number of points plotted",tot

class CatStoreError(Exception):
	pass

class ZoneDoesNotExist(Exception):
	pass
